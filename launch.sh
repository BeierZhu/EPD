# RL Train

## SD1.5
torchrun --master_port=23123 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd15.yaml

### You can replace "/path/to/local/sd3-model" with your local SD3 Medium folder or checkpoint, to avoid downloads.

## SD3-Medium (512x512)
torchrun --master_port=22222 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd3_512.yaml \
    --override model.backend_options.model_name_or_path=/path/to/local/sd3-model

## SD3-Medium (1024x1024)
torchrun --master_port=12345 --nproc_per_node=1 -m training.ppo.launch \
    --config training/ppo/cfgs/sd3_1024.yaml \
    --override model.backend_options.model_name_or_path=/path/to/local/sd3-model

# Sample

## Export the EPD predictor
python -m training.ppo.export_epd_predictor \
    exps/[xxxxxx] \
    --checkpoint checkpoints/policy-step[xxxxxx].pt


## SD1.5
MASTER_PORT=12345 python sample.py \
    --predictor_path exps/[xxxxxx]/export/network-snapshot-export-step000005.pkl \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" \
    --batch 16 \
    --outdir samples/sd15

MASTER_PORT=55551 python sample.py \
    --predictor_path exps/sd15/sd15-best.pkl \
    --prompt-file src/prompts/test.txt \
    --seeds "0-999" \
    --batch 16 \
    --outdir samples/sd15

## SD3-Medium
python sample_sd3.py --predictor exps/sd3-1024/sd3-1024-best.pkl \
    --seeds "0" \
    --outdir samples/sd3 \
    --prompt "A very big apple." \
    --backend-config "{\"model_name_or_path\":\"/path/to/local/sd3-model\"}"

python sample_sd3.py --predictor exps/sd3-512/sd3-512-best.pkl \
    --prompt-file src/prompts/test.txt \
    --seeds "0-9" \
    --max-batch-size 1 \
    --outdir samples/sd3_epd_9_1024_8000 \
    --backend-config "{\"model_name_or_path\":\"/path/to/local/sd3-model\"}"

# Evaluation 
score_all_metrics() {
    local name="$1"
    if [ -z "$name" ]; then
        echo "Usage: score_all_metrics <images_subdir_under_samples>"
        return 1
    fi

    local image_dir="samples/${name}"
    local prefix="${name}"

    mkdir -p results

    python -m training.ppo.scripts.score_clip \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts src/prompts/test.txt \
        --weights weights/clip \
        --output-json "results/${prefix}_clip.json"

    python -m training.ppo.scripts.score_hps \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts src/prompts/test.txt \
        --weights weights/HPS_v2.1_compressed.pt \
        --output-json "results/${prefix}_hps.json"

    python -m training.ppo.scripts.score_aesthetic \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts src/prompts/test.txt \
        --weights weights/sac+logos+ava1-l14-linearMSE.pth \
        --output-json "results/${prefix}_aesthetic.json"

    python -m training.ppo.scripts.score_pick \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts src/prompts/test.txt \
        --weights weights/PickScore_v1 \
        --output-json "results/${prefix}_pick.json"

    python -m training.ppo.scripts.score_imagereward \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts src/prompts/test.txt \
        --weights weights/ImageReward-v1.0.pt \
        --output-json "results/${prefix}_imagereward.json"

    python -m training.ppo.scripts.score_mps \
        --images "${image_dir}" \
        --pattern "**/*.png" \
        --prompts src/prompts/test.txt \
        --weights weights/MPS_overall_checkpoint.pth \
        --output-json "results/${prefix}_mps.json"
}

score_all_metrics ...
