import sys, socket
import time
import threading
import re
import uuid
import torch
from concurrent import futures
import grpc
import service_pb2
import service_pb2_grpc
from threading import Lock, Event
from typing import List, Dict
import os
import os.path
import pickle
from raft_manager import *
import subprocess
import boto3
import shutil

# python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. service.proto

class ReplicaUpdateData: # utility class to wrap up replica information
  def __init__(self, hosted_model_id):
    self.hosted_model_id = hosted_model_id

# A GRPC Servicer that handles the server's actions.
class ServerServicer(service_pb2_grpc.MessageServiceServicer):
  def __init__(self, hosted_model_id, replica_id=None, leader_id=None, replicas={}, out_file=None):
    super().__init__()

    self.hosted_model_id = hosted_model_id

    self.bucket = boto3.resource('s3').Bucket('cs262mj4')

    # Create raft entity, with replica information
    self.raft_manager = RaftManager(replica_id, leader_id, replicas, self.load_raft_data, self.on_raft_data)
    self.out_file = out_file

    if self.out_file and os.path.isfile(self.out_file):
      self.load_file_data()

  def save_file_data(self):
    """
    Save to persistent memory by pickling.
    """
    if not self.out_file:
      return

    with open(self.out_file, 'wb') as file:
      pickle.dump(self.load_raft_data(), file)

  def load_file_data(self):
    """
    Mechanism to load persistent memory.
    """
    if not self.out_file:
      return

    with open(self.out_file, 'rb') as file:
      loaded = pickle.load(file)
      self.hosted_model_id = loaded.hosted_model_id

  def load_raft_data(self):
    """
    Mechanism to load raft data.
    """
    return ReplicaUpdateData(self.hosted_model_id)

  def on_raft_data(self):
    """
    When it gets heartbeat with most recent data, updates replica.
    """
    unloaded = self.raft_manager.latest_data
    self.hosted_model_id = unloaded.hosted_model_id
    self.save_file_data()

  def RaftRequestVote(self, request, context):
    """
    Server-side method for requesting votes from other replicas.
    """
    return self.raft_manager.on_request_vote(request)

  def RaftUpdateState(self, request, context):
    return self.raft_manager.on_heartbeat(request)

  def Get(self, request, context):
    return service_pb2.ModelResponse(success=True, hosted_id=self.hosted_model_id)

  def Merge(self, request, context):
    """
    List accounts, separated by a comma.
    """
    if not self.raft_manager.is_leader():
      return self.raft_manager.leader_stub().Merge(service_pb2.MergeRequest(ckpt_diff_id=request.ckpt_diff_id))

    to_maintain = '/tmp/cs262mj4-' + str(uuid.uuid4()) + '.ckpt'
    to_update = '/tmp/cs262mj4-' + str(uuid.uuid4()) + '.ckpt'
    downloaded_patch = '/tmp/cs262mj4-' + str(uuid.uuid4()) + '.patch'
    self.bucket.download_file(self.hosted_model_id, to_maintain)
    self.bucket.download_file(request.ckpt_diff_id, downloaded_patch)
    subprocess.run(['bspatch4', to_maintain, to_update, downloaded_patch])
    new_file = self.merge_local_files(to_update, to_maintain, 0.3)
    new_file_id = str(uuid.uuid4()) + '.ckpt'
    self.bucket.upload_file(new_file, new_file_id)
    self.hosted_model_id = new_file_id
    return service_pb2.ModelResponse(success=True, hosted_id=self.hosted_model_id)

  def merge_local_files(self, file_a, file_b, ratio):
    model_a = torch.load(file_a)
    model_b = torch.load(file_b)
    theta_a = model_a["state_dict"]
    theta_b = model_b["state_dict"]
    skip_vae = True

    for key in theta_a.keys():
      if skip_vae and "first_stage_model" in key:
        continue
      if "model" in key and key in theta_b:
        theta_a[key] = (1 - ratio) * theta_a[key] + ratio * theta_b[key]
    for key in theta_b.keys():
      if "model" in key and key not in theta_a:
        theta_a[key] = theta_b[key]

    out_file = '/tmp/cs262mj4-' + str(uuid.uuid4()) + '.ckpt'
    torch.save({"state_dict": theta_a}, out_file)
    return out_file

# A server that can start the GRPC servicer on a given port.
class Server():
  def __init__(self, hosted_model_id, replica_id=None, leader_id=None, replicas={}, out_file=None):
    self.servicer = ServerServicer(hosted_model_id, replica_id=replica_id, leader_id=leader_id, replicas=replicas, out_file=out_file)

  def start(self, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_MessageServiceServicer_to_server(self.servicer, server)
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    server.wait_for_termination()

  def force_close(self):
    sys.exit(0)
