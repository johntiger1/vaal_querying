# April 29
Working on getting the environment assumptions oF RL working! In particular: we need to RESET the environment to some original baseline EACH TIME! 

Resetting:
1. Unlabel all the datapoints, OR JUST DUMP THE TRAIN DATALOADER (OK!)
2. Won't work exactly: since we DO implicitly have some state tracking over time...

Even just defining the problem for this question is challenging! Amazing!


Need to do LEARNABLE actions from FIRST principles: batching, environment reset, etc.
(Lmao curriculum learning myself; joint human curriculum learning in RL curriculum learning!)

Also:
we mainly figure out how important the RL reward function is. (ensuring 0 centered in particular).
Also:

THE BIG BET: we will need some sort of experience replay. SOmething like a "standard" environment of 50 candidates to evaluate, and train on, and prep, and then hoping it will generalize as we see more data!

Stagewise RL; as we get more samples, we get exponentially more sequences. Lots of nice FUNDAMENTAL ways to motivate this problem!

# April 28
Got the policy gradient (correct implementation working).


    # one other thing: doing a value network, or framing it as a regression problem
            # RL seems like the wrong formulation to do AL.
            
            
                # try and predict directly in the data space?
    # try and graph where in the dataset it does it as well.
    # in this case, we would again need some fix of the policy gradient.
    # we can no longer just do an easy cross entropy
    # instead, we would be operating more in the regime of .
    # this is a nice analysis on problems of this type!

    # just about rotating and sculpting to your particular area you want
    # ultra few shot learning with fixed dimension, horizon!
    # contributions: to the field of meta/few shot learning using an active learning with reinforcement learning approach
    # keep on layering intersections, until you get the particular area you want.
    # even the approach of doing few shot learning, using active learning is pretty novel IMO




    # we would be trying to essentially do q learning on the choice of datapoint. but make sure you pick in the data space (not action, but continuous choice of the datapoint)
    # the key is really then, trying to do X
    # we could literally do an entire course of lin alg during the break!

    # really, digging into the problems of policy gradient

    # now let's graph the unlabelled dataset
