import torch
import time
import wandb
from tqdm import tqdm
import numpy as np
import accelerate
from distrl.models import timeout
use_tars = True
def add_trajectory_reward(trajectory):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_reward = np.sum([d["reward"] for d in trajectory])
    for d in trajectory:
        d.update({"trajectory_reward": trajectory_reward})
    return trajectory

def add_mc_return(trajectory, gamma = 0.95):
    """
    
    add trajectory reward to the dict of each interaction
    

    """
    trajectory_rewards = np.array([d["reward"] for d in trajectory]).reshape(1, -1)
    gamma_row = np.cumprod(np.ones((1, trajectory_rewards.shape[1]))*gamma)
    gamma_matrix = np.triu(gamma_row.reshape(1, -1 )/ gamma_row.reshape(-1, 1))
    mc_returns = np.sum(trajectory_rewards*gamma_matrix, axis = 1)
    for d, mc in zip(trajectory, mc_returns):
        d.update({"mc_return": mc})
    return trajectory

def add_log_prob_for_deterministic_actions(trajectory):
    """
    Add log probability for deterministic actions to each interaction in the trajectory.
    
    Parameters:
        trajectory (list of dicts): The trajectory data where each entry is a dictionary representing a step.
    
    Returns:
        list of dicts: The trajectory with added 'log_prob' values, set to 0 for deterministic choices.
    """
    # Since actions are deterministic, set log_prob to 0 (log(1))
    max_prob = 0.99
    for step in trajectory:
        step["log_prob"] = np.log(max_prob) # log(1) because it's a deterministic policy
    
    return trajectory

def batch_interact_environment(agent, env, num_trajectories,\
        accelerator, post_f = lambda x: x, use_tqdm = True, decode_f = lambda x: x, gamma = 0.95, iter=0, use_wandb=True):
    """
    in a bacthed way, interact with the environments  to get a list of trajectories
    [[{"observation":, "next_observation":, "reward":, "done":},...],...]
    post_f: function to add additional attributes to the trajectory
    """
    # broadcast the batch size
    print(f">>> Start interacting: {iter}...")
    bsize = torch.Tensor([0,]).to(accelerator.device)
    if accelerator.is_main_process:
        bsize[0] = env.bsize
    accelerate.utils.broadcast(bsize)
    bsize = int(bsize.item())
    all_trajectories = []
    if accelerator.is_main_process:
        if hasattr(agent, "critic"):
            env.feature_extractor.model = env.feature_extractor.model.to(env.device)
            # agent.critic.to("cpu")
    for num_t in tqdm(range(num_trajectories//bsize), disable=not use_tqdm):
        if accelerator.is_main_process:
            env.emulator_group_offset = iter * num_trajectories + num_t * bsize
        for _ in range(3):
            try:
                done = False
                trajectories = [[] for _ in range(bsize)]
                #handle the case where the reset fails and timeouts
                reset_success = torch.Tensor([False,]).to(accelerator.device)
                while not all(reset_success):
                    try:
                        if accelerator.is_main_process:
                            with timeout(seconds=500):
                                batch_obs = env.reset()
                            #the observation space is now a tuple of (text, image)
                            if type(batch_obs[0]['image_feature']) == torch.Tensor:
                                batch_img = [obs["image_feature"] for obs in batch_obs]
                            else:
                                batch_img = ["Image feature is not a tensor" for _ in range(bsize)]
                            if env.feature_extractor is not None:
                                # colorful_print("autoui has critic, so batch_obs being refractored", "red")
                                if not use_tars:
                                    batch_obs = [obs["prompt"] for obs in batch_obs]
                                else:
                                    print("carl keep batch_obs")
                                    batch_obs = [obs["msg_q"] for obs in batch_obs]
                            reset_success[0] = True
                        accelerate.utils.broadcast(reset_success)
                        break
                    except Exception as e:
                        print(f"[!] Error in environment reset: {e}")
                        accelerate.utils.broadcast(reset_success)
                        continue

                # run task
                batch_done = torch.Tensor([False,]*bsize).to(accelerator.device)
                accelerate.utils.broadcast(batch_done)
                steps = 0
                start = None
                while not all(batch_done):
                    # TODO: make it log
                    if accelerator.is_main_process:
                        end = time.time()
                        steps += 1
                        print("Step:", steps)
                        if start is not None:
                            print("Time:", end-start)
                        start = end
                        if use_wandb:
                            info = {
                                "Step": steps,
                                "Walltime": end
                            }
#                            wandb.log(info)
                        
                        if env.feature_extractor is not None:
                            with torch.no_grad():
                                action = agent.get_action(batch_obs, torch.cat([i.unsqueeze(0) for i in batch_img], dim = 0))
                                #log_prob = agent.get_log_prob(batch_obs, torch.cat([i.unsqueeze(0) for i in batch_img], dim = 0), action).sum(dim=1).flatten()
                        else:
                            with torch.no_grad():
                                action = agent.get_action(batch_obs, None)
                                log_prob = agent.get_log_prob(batch_obs, None, action).sum(dim=1).flatten()
                        
                        # import IPython; IPython.embed(); exit(1)
                        with timeout(seconds=5*60):
                            batch_return = env.step(decode_f(action))
                        # batch_return = env.step(decode_f(action))
                        # import IPython; IPython.embed()
                        for i,result in zip(range(bsize), batch_return):
                            if result is None:
                                batch_done[i] = True
                                continue
                            obs_dict, r, done, penalty = result
                            next_img = obs_dict["image_feature"]
                            next_obs = obs_dict["prompt"]
                            if not hasattr(agent, "critic"):
                                trajectories[i].append({"observation": batch_obs[i], \
                                    "next_observation": next_obs, \
                                    "image_features": None, \
                                    "image_path": obs_dict["image_path"], \
                                    "next_image_features": None, \
                                    "task": obs_dict["task"],\
                                    "reward": r, \
                                    "penalty": penalty, \
                                    "done": done, \
                                    "action": action[i], \
                                    "log_prob": log_prob[i]})
                                batch_obs[i] = obs_dict
                            else:
                                trajectories[i].append({"observation": batch_obs[i], \
                                    "next_observation": next_obs, \
                                    "image_features": batch_img[i].cpu().numpy(), \
                                    "image_path": obs_dict["image_path"], \
                                    "next_image_features": next_img.cpu().numpy(), \
                                    "task": obs_dict["task"],\
                                    "reward": r, \
                                    "penalty": penalty, \
                                    "done": done, \
                                    "action": action[i], \
                                    #"log_prob": log_prob[i]
                                                       })
                                if not use_tars:
                                    batch_obs[i] = next_obs
                                else:
                                    batch_obs[i] = obs_dict["msg_q"]
                            batch_img[i] = next_img
                            batch_done[i] = done
                    accelerate.utils.broadcast(batch_done)
                    # print("waiting for everyone")
                    # accelerator.wait_for_everyone()
                    # obs = next_obs
                if accelerator.is_main_process:
                    print(trajectories[0][-1]["next_observation"])
                    all_trajectories += [post_f(add_mc_return(add_trajectory_reward(trajectory), gamma=gamma))\
                                        for trajectory in trajectories]
                break
            except Exception as e:
                print(f"Error in environment interaction")
                import traceback
                print(traceback.format_exc())
                print(e)
                continue
    if accelerator.is_main_process:
        if env.feature_extractor is not None:
            env.feature_extractor.model = env.feature_extractor.model.to("cpu")
            # if hasattr(agent, "critic"):
            #     agent.critic.to(agent.device)
        
    return all_trajectories
