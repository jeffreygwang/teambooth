**Background**

Dreambooth is a generative AI model based on stable diffusion that enables photorealistic generation of subjects from a given text prompt in different contexts. Given just a few novel sample images that represent a given subject, Dreambooth models allow users to generate their subject in any context using only natural language. Since its release in mid-2022, Dreambooth-enabled generation has taken the world by storm—which naturally prompts the question: how can I do this too?

To the best of the authors' knowledge, there is currently no "drag and drop" version of Dreambooth. In fact, the only way we found to access it is via a few costly subscription plans, the [cheapest](https://getimg.ai/pricing) of which is a $12 monthly subscription for two generations.  On some level, despite how magical Dreambooth generations can be, its lack of accessibility is no surprise: fine-tuning is incredibly expensive and time-consuming. Offering such a free service is simply not scalable because the unit economics do not make sense.

Nevertheless, there is one important fact of Dreambooth and many generative AI models in general: that *generation* of a given subject in some context is far cheaper than *fine-tuning* that subject into the model. In other words, once a subject has been "trained in", generating that subject is a cheap operation. This aligns well with what we envision to be a likely end user of a Dreambooth application: someone who wishes to fine-tune on just a few subjects (e.g. their dog, their friend, etc.) but then generate and share many times. Pithily: light on fine-tuning, heavy on generation, and social.

As a result, a federated learning approach can make Dreambooth (and other compute-intensive, personalized generative AI models) much more accessible. While federated learning's primary application has traditionally been in privacy contexts, the other aspect—distributing the actual training of the model—makes a lot of sense here. Users take on up-front compute for fine-tuning, while their eventual central server model can be shared with others. 

**Engineering Overview**

One major engineering challenge was figuring out how to create a "federated" model when interacting with a Dreambooth model is time- and compute-intensive. The base "image" for Dreambooth before fine-tuning, for instance, is over 3 gigabytes. We felt that sending these models over the wire would be prohibitively time-consuming and prone to failure. Additionally, since most existing users of these models do not have GPU's on their local machines; they likely use some cloud GPU provider to access their compute.

As such, the first major engineering decision we made was to make Teambooth AWS-native. In AWS, all entities stored in S3, the AWS-native storage solution, are referenced by specific ID's. As a result, rather than sending enormous models over the wire, we simply send small packets of ID information—something for which we used gRPC. While there is still plenty of hair in consensus here, it is far simpler than maintaining consistency with gigabyte-sized models.

By using this simplification, keeping the PR and SR models in sync reduces to keeping a log updated between PR and SR replicas with the right "current global model." In our application, we use Raft to do so. A brief description is offered below:
- Raft divides the distributed consensus problem into three subproblems: leader election, log replication, and safety.
- In Raft, there is a leader who is responsible for managing the replication of the log. Other nodes are followers, which replicate the log. This is the primary/secondary or leader/follower approach to consensus.
- If the leader fails, a new leader is elected using a randomized timeout mechanism. In brief, every non-leader replica starts as a follower, sending heartbeat pings to the leader at random intervals. Whenever a replica does not get a heartbeat from a leader, it becomes a _candidate_ for the leader, sending out requests for votes to other replicas; if a majority of other nodes get its candidacy request and vote for it, it becomes the leader.
- The leader receives commands from clients and appends them to its log. The leader then replicates its log to its followers, who in turn replicate it to other followers.
- Raft ensures safety by enforcing a rule that a log entry can only be committed if it has been replicated on a majority of the nodes.

One asterisk of our implementation is that all server-server communication is done via the heartbeats, since the data packages are small (just packets containing e.g. ID info). Other than that, by reduction to an AWS-specific platform, we simplify the complex problem of maintaining consistency into a standard version of Raft.

As noted in the beginning, in our implementation, the PR handles all MERGEs and the SR machines handle all FINE-TUNE and INFER requests; these amount to serving GET requests of the latest model version. We currently handle this in a simple way; on boot, a client must enter the IPs of all PR/SR machines, and processing of where to send requess is done on the client side. In a production application, we would implement a reverse proxy (which could also be distributed, to avoid a single point of failure) on the server side to handle this.

**Use Cases and Benefits: Who Wins?**

Given the above implementation, the final use scenario with the servers up would look as follow:
- Different users that want to FINE-TUNE can request the global model and train on specific subjects.
- As fine-tuning finishes for users at different times (because of compute capabilities), they send the model back to the primary server, which runs MERGE on those models into the global model.
	- Any time the PR finishes a merge, it propagates that new model ID to secondary replicas.
- Any time a client wishes to INFER, they send a request to an SR, which serves the most recent model ID.

We believe the clients and the server wins here. Clients win because they can run Dreambooth in a fairly accessible way, avoiding exorbitant pricing from sharky online services. The server wins because they can offload most compute to the client, making a sharable "Dreambooth-as-a-service" application immensely more scalable.

**Unit Economics**

Testing this application was difficult because of the intense compute. We had GRPC unit tests and tested the fault-tolerance of our application. See `Testing/Resilience` in the second part of our report for more on that. As such, most of our experiments for this report were on proof-of-viability, as we tested the feasability and appeal of this approach from a unit economics perspective.

Using `g3.xlarge` [instances](https://aws.amazon.com/ec2/instance-types/g3/), which are NVIDIA Tesla M60 GPUs with 2048 parallel processing cores and 8-32 GiB of GPU memory, fine-tuning to good quality takes on the order of 10 minutes to 2 hours, depending on hyperparameter configurations and machine type (we used 800 fine-tuning iterations). Generation of an image, on the other hand, takes 5-60 seconds: anywhere from 1-3 orders of magnitude less. As such, given that these machines run from $0.75/hr to $4.56/hr, a single generation costs about $1-2 on average for the client. See table below:

| Instance Type | Time for Fine-Tuning | Cost/Hr ($) | Approx Cost per Fine-Tune |
| -------- | -------- | -------- | -------- |
|   g3s.xlarge (8GB GPU RAM)    |   ~2 hours     |      0.75    |       2.25   |
|   g3.4xlarge   (8GB GPU RAM)     |     ~1 hour     |    1.14      |     1.14    |
|     g3.8xlarge  (16GB GPU RAM)    |     ~30 mins     |      2.28    |    1.14    |
|     g3.16xlarge    (32GB GPU RAM)  |    ~10 mins      |      4.56    |     0.46   |

> One detail we omit here is that running on smaller instances (with limited GPU RAM) often requires parallelization and memory-reduction techniques like [gradient checkpointing](https://huggingface.co/docs/transformers/v4.18.0/en/performance) and [DeepSpeed](https://github.com/microsoft/DeepSpeed), hence why the times do not scale linearly with e.g. GPU RAM.

Once a model has been fine-tuned, generating Dreambooth context variations is similarly affordable. For generation, we used 50 diffusion steps in generation on a g3.16xlarge, which took approximately 10 seconds. With these unit economics, one could generate approximately 80 different Dreambooth instances per dollar of compute. All in all, this is more cost efficient for clients than existing models.

The real winnings come for the servers. By offloading compute to clients, much more is possible. Using the most powerful instance here (`g3.16xlarge`), a single client-server model could service at maximum 144 (6 * 24) fine-tunes per day, at $109.44 in compute.

In our testing, we ran merges with the smallest instance, a `g3s.xlarge`, where they take approx 2-5 minutes. Now considering a distributed paradigm, assuming an average of ~3.3 minutes per merge, the primary server could handle 3x the number of clients at 0.75/4.56 = 16 percent of the cost. The secondary replicas would only need to serve GET requests, so they could be AWS's cheapest machines (which run at approximately 1 cent/hour). All in all, this comes out to be 18x more efficient (on a number-of-daily-clients-per-cent basis) for servers vs. the central server-client model—savings which do not take into account increased uptime and fault tolerance from distribution.

One alternative to merging into a global model, given the "concept drift" and "vanishing" effects, is to simply store all client models. One issue with this paradigm is that each fine-tuned model is ~3GB in size, which scales quickly with users. However, from a compute perspective, it is even more efficient than our model (which needs merges) and similarly cost-effective at a small scale given how cheap storage is.
