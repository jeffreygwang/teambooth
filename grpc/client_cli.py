import grpc
import sys
import service_pb2
import service_pb2_grpc
import time
import uuid
import subprocess
import boto3

DISP_MSG = "Select from your options:" \
        "\n Enter 0 to get the current model." \
        "\n Enter 1 merge your changes into the model." \
        "\n Enter 2 for instructions on training and inference." \
        "\n Enter Q to exit the application. \n\nYour choice: "

bucket = boto3.resource('s3').Bucket('cs262mj4')

class ClientCli():
  def __init__(self, servers):
    # A `MessageServiceStub` object from GRPC for client interactions.
    self.client = None

    # Alternate servers to try if one goes down.
    self.servers = servers

    # Path to current model file.
    self.current_model_path = None

  # Starts the main loop for listening to user inputs and sending responses.
  def user_loop(self):
    valid_responses = ["H", "0", "1", "Q"]

    firstTime = True
    while True:
      response = None

      if firstTime:
        firstTime = False
        response = self.user_query()
      else:
        disp_msg = "Take your next action. Press H to get the directions again. "
        response = self.user_query(disp_msg)

      if response == "H":
        firstTime = True
        continue
      elif response == "0":
        self.get_model()
      elif response == "1":
        new_model_path = self.user_query("Enter the path to the newly-trained model.")
        self.update_model(new_model_path)
      elif response == "2":
        print(
"""To train, use a series of commands such as:
export MODEL_NAME="/tmp/model-path.ckpt" # or "CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR=instance-1
export OUTPUT_DIR=out-1
accelerate launch --mixed_precision="fp16" train_dreambooth.py \\
--pretrained_model_name_or_path=$MODEL_NAME \\
--instance_data_dir=$INSTANCE_DIR \\
--output_dir=$OUTPUT_DIR \\
--instance_prompt="a photo of [your pompt here]” \\
--resolution=512 \\
--train_batch_size=1 \\
--sample_batch_size=1 \\
--gradient_accumulation_steps=1 --gradient_checkpointing \\
--learning_rate=5e-6 \\
--lr_scheduler="constant" \\
--lr_warmup_steps=0 \\
--num_class_images=200 \\
--max_train_steps=800"

It may be necessary to perform a conversion:
python3 ../../scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path=[received ckpt file] --dump_path=[your model path]

After, you may also want to run:
python3 ../../scripts/convert_diffusers_to_original_stable_diffusion.py --model_path=[your model path] --checkpoint_path=[ckpt out file]

To infer, run

python3 infer.py
"""
        )
      elif response == "Q":
        break

  # Starts the GRPC client and begins the `user_loop`.
  def main(self):
    response: str = input("Welcome to the chat app! Press 1 to continue. Press 0 to quit. ")
    while response != "1" and response != "0":
      response: str = input("Press 1 to continue. Press 0 to quit. ")
    if response == "0":
      sys.exit(0)

    channel = grpc.insecure_channel(self.servers[0])
    self.client = service_pb2_grpc.MessageServiceStub(channel)
    self.user_loop()

  def try_new_channel(self):
    print(f'Updating active server. Popped {self.servers.pop(0)} - now server is {self.servers[0]}')
    channel = grpc.insecure_channel(self.servers[0])
    self.client = service_pb2_grpc.MessageServiceStub(channel)

  def user_query(self, msg: str=DISP_MSG) -> str:
    return input(msg)

  # Fetches current model.
  def get_model(self):
    try:
      response = self.client.Get(service_pb2.GetRequest())
      if response.hosted_id:
        to_save = '/tmp/cs262mj4-' + str(uuid.uuid4()) + '.ckpt'
        print('Downloading model: ' + response.hosted_id)
        bucket.download_file(response.hosted_id, to_save)
        print('Saved model to: ' + to_save)
        self.current_model_path = to_save
      else:
        print('Error: No model available.')
        return None
    except:
      print("Error: Unable to connect to server. Please try again.")
      self.try_new_channel()

  # Updates the model by computing a diff.
  def update_model(self, new_model_path):
    try:
      if not self.current_model_path:
        print('Error: Missing current model.')
        return

      patch_location = '/tmp/cs262mj4-' + str(uuid.uuid4()) + '.patch'
      subprocess.run(['bsdiff4', self.current_model_path, new_model_path, patch_location])
      new_file_id = str(uuid.uuid4()) + '.patch'
      bucket.upload_file(patch_location, new_file_id)
      response = self.client.Merge(service_pb2.MergeRequest(ckpt_diff_id=new_file_id))
      if response.hosted_id:
        to_save = '/tmp/cs262mj4-' + str(uuid.uuid4()) + '.ckpt'
        bucket.download_file(response.hosted_id, to_save)
        print('Saved model.')
        self.current_model_path = to_save
      else:
        print('Error: No model available.')
        return None
    except:
      print("Error: Unable to connect to server. Please try again.")
      self.try_new_channel()

