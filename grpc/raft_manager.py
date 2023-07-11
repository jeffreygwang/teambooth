import uuid
import time
import random
import grpc
import service_pb2
import service_pb2_grpc
import threading
import pickle

class ReplicaInformation:
  def __init__(self, replica_id, url):
    self.id = replica_id
    self.url = url

  def stub(self):
    connection = grpc.insecure_channel(self.url)
    return service_pb2_grpc.MessageServiceStub(connection)

class RaftManager:
  def __init__(self, replica_id=None, leader_id=None, replicas={}, load_data=None, on_new_data=None):
    # Set Raft information
    self.id = replica_id or str(uuid.uuid4()) # replica id
    self.replicas = replicas                  # {replica_id: ReplicaInformation}
    self.leader_id = leader_id or self.id     # what this server believes is the leader id (could be self)
    self.term = 0                             # raft term (monotonically increasing)
    self.load_data = load_data                # init data loading dictionary
    self.on_new_data = on_new_data            # method for getting new data
    self.last_heartbeat = time.time()         # time of last heartbeat
    self.election_timeout = 2 + 2 * random.random() # election timeout

    self.send_heartbeat()
    self.leader_check_interval()

  def is_leader(self):
    """
    Check if leader.
    """
    return self.id == self.leader_id

  def leader_stub(self):
    """
    Stub for leader. 
    """
    return self.replicas[self.leader_id].stub()

  def send_heartbeat(self):
    """
    Send heartbeats to replicas. 
    """
    t = threading.Timer(0.5, self.send_heartbeat)
    t.daemon = True
    t.start()

    if self.is_leader():
      data_string = pickle.dumps(self.load_data())
      for rid in self.replicas:
        print('[raft] Sending heartbeat to replica.')
        try:
          self.replicas[rid].stub().RaftUpdateState(service_pb2.RaftUpdateStateRequest(replica_id=self.id, data=data_string))
        except Exception as e:
          print(f"[raft] Exception raised: {e}")
          pass

  def on_heartbeat(self, request):
    """
    When receiving a heartbeat, update state. Heartbeats contain data.
    """
    if request.replica_id is self.leader_id:
      self.last_heartbeat = time.time()
      self.latest_data = pickle.loads(request.data)
      print(f'[raft] Received heartbeat from primary replica {request.replica_id}')
      if self.on_new_data:
        self.on_new_data()

    return service_pb2.EmptyResponse(success=True)

  def on_request_vote(self, request):
    """
    On request to vote, execute this behavior.
    """
    # Documentation
    print(f"[raft] Current Term: {self.term}")
    print(f"[raft] Request Term: {request.term}")

    # If term greater, vote yes.
    if request.term > self.term:
      self.term = request.term
      self.leader_id = request.candidate_id
      print(f"[raft] Requested ID: {request.candidate_id}")
      print('[raft] Received vote request, voting yes')
      return service_pb2.RaftRequestVoteResponse(vote=True)
    else:
      print('[raft] Received vote request, voting no')
      return service_pb2.RaftRequestVoteResponse(vote=False)

  def leader_check_interval(self):
    """
    Main interval execution behvior in Raft.
    """
    t = threading.Timer(self.election_timeout, self.leader_check_interval)
    t.daemon = True
    t.start()

    # Regularly scheduled execution intervals
    if time.time() - self.last_heartbeat > self.election_timeout and not self.is_leader():
      print('[raft] Heartbeat timeout')
      self.term += 1
      print(f"[raft] New Term: {self.term}")

      # Every replica votes for itself
      votes = 1
      votes_yes = 1

      for rid in list(self.replicas):
        print('[raft] Sending vote request to replica')
        try:
          response = self.replicas[rid].stub().RaftRequestVote(service_pb2.RaftRequestVoteRequest(term=self.term, candidate_id=self.id))
          if response:
            votes += 1
            if response.vote:
              print(f"[raft] Replica {rid} voted yes.")
              votes_yes += 1
            else:
              print(f"[raft] Replica {rid} voted no.")

        except:
          print('[raft] Removing replica key:', rid)
          del self.replicas[rid]

      if votes_yes > votes / 2:
        print(f"[raft] Becoming leader : current term {self.term}")
        self.last_heartbeat = time.time()
        self.leader_id = self.id

