import sys
sys.path.append('./')

import torch
from OneLLM.model.meta import MetaModel
import numpy as np
import torch.distributed as dist
from OneLLM.data.conversation_lib import conv_templates
from fairscale.nn.model_parallel import initialize as fs_init
from OneLLM.util.misc import setup_for_distributed
from OneLLM.util.misc import default_tensor_type

def init_onellm():
    pretrained_path = "./cache/consolidated.00-of-01.pth"
    # mp.set_start_method("spawn")
    dist.init_process_group(
        backend="nccl", rank=0, world_size=1,
        init_method=f"tcp://127.0.0.1:23560")
    fs_init.initialize_model_parallel(1)
    torch.cuda.set_device(0)
    torch.manual_seed(1)
    np.random.seed(1)
    # set the print behavior.
    # setup_for_distributed(True)

    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16
    }['fp16']
    with default_tensor_type(dtype=target_dtype, device="cuda"):
        model = MetaModel("onellm", "OneLLM/config/llama2/7B.json", None, "OneLLM/config/llama2/tokenizer.model")
       
    print("Loading pretrained weights ...")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)
    print("load result:\n", msg)
    model.half().cuda()
    model.eval()
    print(f"Model = {str(model)}")
    return model, target_dtype

def inference_onellm(model, target_dtype, images, modal=['image']):
    if 'imu' in modal:
        inps = ['Describe the motion.'] * len(images)
    if 'audio' in modal:
        inps = ['Provide a one-sentence caption for the provided audio.'] * len(images)
        # inps = ['Provide a one-sentence action description for the provided audio.'] * len(images)
    if 'image' in modal:
        inps = ['Describe the scene.'] * len(images)
    images = images.cuda().to(target_dtype)
    prompts = []
    for inp in inps:
        conv = conv_templates["v1"].copy()        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompts.append(conv.get_prompt())

    with torch.cuda.amp.autocast(dtype=target_dtype):
        responses = model.generate(prompts, images, 128, temperature=0.1, top_p=0.75, modal=modal)
        outputs = []
        for response, prompt in zip(responses, prompts):
            response = response[len(prompt):].split('###')[0]
            response = response.strip()
            outputs.append(response)
    return outputs

if __name__ == "__main__":

    model, target_dtype = init_onellm()
    nsamples = 400
    imus = torch.zeros(1, 6, nsamples) + 0.5
    imu_names = ['dummpy.npy']
    imu_ids = torch.tensor([0])

   
    results = inference_onellm(model, target_dtype, imus, modal=['imu'])
    
    for result in results:
        print(result)
    