import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Multi-Agent-Semantic-Exploration')

    # General
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('-d', '--dump_location', type=str, default="./tmp")
    parser.add_argument('--exp_name', type=str, default="exp1")
    parser.add_argument('-v', '--visualize', type=int, default=0)
    parser.add_argument('--print_images', type=int, default=0)

    # Env
    parser.add_argument('-fw', '--frame_width', type=int, default=640)
    parser.add_argument('-fh', '--frame_height', type=int, default=480)
    parser.add_argument("--task_config", type=str, default="multi_objectnav_hm3d.yaml")
    parser.add_argument('--hfov', type=float, default=79.0)

    # Algo
    parser.add_argument('--agent', type=str, default="sem_exp")
    parser.add_argument('--num_local_steps', type=int, default=25)
    parser.add_argument('-n', '--num_processes', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--map_size_cm', type=int, default=2400)
    parser.add_argument('--map_height_cm', type=int, default=130)
    parser.add_argument('--sem_threshold', type=float, default=0.85)
    parser.add_argument('--num_agents', type=int, default=2)

    # Navigation mode
    parser.add_argument('--nav_mode', type=str, default="gpt",
                        choices=['nearest', 'co_ut', 'fill', 'gpt'])
    parser.add_argument('--fill_mode', type=int, default=0)

    # ===== MiniCPM-o local server config (FastAPI /infer) =====
    parser.add_argument('--vlm_backend', type=str, default="minicpm",
                        choices=["minicpm"], help="VLM backend used by chat_utils")
    parser.add_argument('--minicpm_url', type=str, default="http://127.0.0.1:8001/infer",
                        help="FastAPI endpoint for MiniCPM-o-2.6 /infer")
    parser.add_argument('--minicpm_timeout_s', type=float, default=120.0)
    parser.add_argument('--minicpm_retries', type=int, default=3)
    parser.add_argument('--minicpm_session_id', type=str, default="conav_default",
                        help="Server-side session_id for multi-turn memory (optional)")
    parser.add_argument('--minicpm_use_one_shot', type=int, default=0,
                        help="1: call /set_one_shot once per run (if implemented in chat_utils)")
    parser.add_argument('--minicpm_one_shot_answer', type=str, default="I will output JSON only.",
                        help="Assistant exemplar for /set_one_shot (if enabled)")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
