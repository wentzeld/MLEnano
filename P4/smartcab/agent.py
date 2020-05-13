import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        # this is our learning rate which should be of the form 1/t t being our deadline perhaps as it is a counter, although it will go negative
        self.debug = 2 #0 prints nothing, 1 prints deadlines net reward, positive reward and negative reward sums, 2 prints more details
        self.myTimer = 0.0
        self.originalalpha = "1overt" #this can be a float or make it "1overt" in which case learning declines as time passes
        self.originalepsilon = 0.9
        self.originalgammadiscount = 0.9
        self.originalnumberfullyrandomsims = 90
        self.alpha = self.originalalpha
        self.epsilon = self.originalepsilon
        self.numberfullyrandomsims = self.originalnumberfullyrandomsims
        self.gammadiscount =self.originalgammadiscount
        self.QhatTable = {}
        self.simnumber = 0
        self.randomsims = 0
        self.nonrandomsims = 0
        self.thisrunrandomsims = 0
        self.thisrunnonrandomsims = 0
        self.netreward = 0
        self.posreward = 0
        self.negativereward = 0
        self.thisrunnetreward = 0
        self.thisrunposreward = 0
        self.thisrunnegativereward = 0

        self.filenam = ''
        self.outputfile = 'Q_out_alpha_'+str(self.alpha) + '_epsilon_'+ str(self.epsilon)+'_gamma_' +str(self.gammadiscount) +'_randomsims_' + str(self.numberfullyrandomsims) +'.txt'
        self.filenam = open(self.outputfile, 'wt')
        self.filenam2 = ''
        self.outputfile2 = 'LearningRate_alpha_'+str(self.alpha) + '_epsilon_'+ str(self.epsilon)+'_gamma_'+ str(self.gammadiscount) +'_randomsims_' + str(self.numberfullyrandomsims)+ '.txt'
        self.filenam2 = open(self.outputfile2, 'wt')


        # build dictionary within a dictionary within a dictionary
        # first dictionary will be our state and the second will be our action
        # we then assign 0.0 to our state:action dictionary
        # state is lights:oncoming traffic:traffic from right:traffice from left:waypoointaction suggestion

        for lightcolour in ['red','green']:
            for oncomingdirection in ['forward','right','left','None']:
                for rightdirection in ['forward','right','left','None']:
                    for leftdirection in ['forward','right','left','None']:
                        for waypointaction in ['forward','right','left','None']:

                                    statetoinsert = lightcolour+','+oncomingdirection+','+rightdirection+','+leftdirection+','+waypointaction
                                    if statetoinsert in self.QhatTable :
                                        a = 1
                                    else:
                                        self.QhatTable[statetoinsert] = {}
                                        self.QhatTable[statetoinsert]['forward'] = 0.0
                                        self.QhatTable[statetoinsert]['right'] = 0.0
                                        self.QhatTable[statetoinsert]['left'] = 0.0
                                        self.QhatTable[statetoinsert][None] = 0.0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.myTimer = 0.0
        self.alpha = self.originalalpha
        self.epsilon = self.originalepsilon
        self.gammadiscount = self.originalgammadiscount
        self.simnumber += 1
        self.thisrunrandomsims = 0
        self.thisrunnonrandomsims = 0
        self.thisrunnetreward = 0
        self.thisrunposreward = 0
        self.thisrunnegativereward = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.myTimer += 1
        if self.originalalpha == "1overt":
            self.alpha = float(1/self.myTimer) #here we update our learning function based on time lapsed
        # I would like to stop using random actions after the 50th run
        self.epsilon = max(0.0,self.epsilon-self.myTimer/(2**self.numberfullyrandomsims/min(2**self.numberfullyrandomsims,2**self.simnumber)))
        destination = self.env.agent_states[self]['destination']
        location = self.env.agent_states[self]['location']
        # Current State
        myState = getdictinput(inputs,self.next_waypoint)
        self.state = myState
        # based on the max expected utility of next action we need to decide on action, with some randomness
        myPrevState = myState
        myaction = getdirection(random.randint(1,4))
        #now we need to implement epsilon greedy
        randomchoice = False
        epsRand = random.randint(0,100)/100.0
        if epsRand <= self.epsilon:
            # if we select randomly we have already done so, no need to repeat
            # due to the way we have defined our states, we can really end up in any of the above states
            #light could be green or red, could be cars or nocoars, waypoint could go either way
            # so lets loop through our whole state space?
            maxQ  = getMaxQ(self.QhatTable,myState,myaction)
            randomchoice = True
            if self.debug == 2:
                print("Simulation number: " + str(self.simnumber))
                print("Random Decision Q for "+ str(myaction) + " is " + str(maxQ))
        else:
            maxQ = 0.0
            # can't do random selection so we have to go through each of the possible actions we can choose
            for ii in [1,2,3,4]:
                tempDirection = getdirection(ii)

                maxQTemp = getMaxQ(self.QhatTable,myState,tempDirection)
                if self.debug == 2:
                    print("Simulation number: " + str(self.simnumber))
                    print("Show Q for all actions")
                    print("Learned Decision Q for "+ str(tempDirection) + " is " + str(maxQTemp))
                if maxQTemp > maxQ:
                    myaction = tempDirection
                    maxQ = maxQTemp


        # TODO: Select action according to your policy
        action = myaction
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.previousstate = myState
        self.thisrunnetreward += reward
        self.netreward += reward
        if reward >=0:
            self.thisrunposreward += reward
            self.posreward += reward
        else:
            self.thisrunnegativereward += reward
            self.negativereward += reward
        if randomchoice:
            self.randomsims += 1
            self.thisrunrandomsims += 1
        else:
            self.nonrandomsims += 1
            self.thisrunnonrandomsims += 1


        # TODO: Learn policy based on state, action, reward
        # if we are not done then update matrix otherwise it may skew our driving in the last step?
        if self.debug == 2:

            print("Maximum Q for "+ str(action) + " is "+ str(maxQ))
            print("Next Waypoint according to planner: " + str(self.next_waypoint))
            print("Our Next Waypoint according to code: " + str(myaction))
            print("Our reward: "+str(reward))
            print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {} \n".format(deadline, inputs, action, reward)  # [debug]

        if self.debug ==1:
            print("Simulation number: " + str(self.simnumber))

            print("Total Number of decisions, overall: "+str(self.nonrandomsims + self.randomsims))
            print("Overall rewards")
            print("Net Reward:      " + str(self.netreward))
            print("Positive Reward: " + str(self.posreward))
            print("Negative Reward: " + str(self.negativereward))
            print("Total Number of decisions, this simulation: "+str(self.thisrunnonrandomsims + self.thisrunrandomsims))
            print("This Simulation rewards")
            print("Net Reward:      " + str(self.thisrunnetreward))
            print("Positive Reward: " + str(self.thisrunposreward))
            print("Negative Reward: " + str(self.thisrunnegativereward))
            print("Deadline: "+str(deadline))

        if self.env.done != True:
            self.QhatTable[self.previousstate][action] = (1.0-self.alpha)*self.QhatTable[self.previousstate][action] + self.alpha*(float(reward) + float(self.gammadiscount * maxQ))
        else:
            printLearningRate(self.filenam2, deadline, self.simnumber,str(self.thisrunnonrandomsims + self.thisrunrandomsims), self.netreward, self.posreward, self.negativereward, self.thisrunnetreward, self.thisrunposreward,  self.thisrunnegativereward,self.originalalpha,self.originalepsilon,self.originalgammadiscount,self.originalnumberfullyrandomsims)
            printDict(self.filenam, self.QhatTable, self.simnumber)

        print("\n")


def getMaxQ(Dict,statetoinsert,myaction):
    maxQ = float(Dict[statetoinsert][myaction])
    return maxQ


def printDict(f, DictToPrint, simnumber):
    t = ','
    for lightcolour in ['red','green']:
        for oncomingdirection in ['forward','right','left','None']:
            for rightdirection in ['forward','right','left','None']:
                for leftdirection in ['forward','right','left','None']:
                    for waypointaction in ['forward','right','left','None']:
                        statetoinsert = lightcolour+','+oncomingdirection+','+rightdirection+','+leftdirection+','+waypointaction
                        writenext = str(simnumber) + t + str(statetoinsert) +t+'forward'+t+str(DictToPrint[statetoinsert]['forward'])+'\n'
                        f.write(writenext)
                        writenext = str(simnumber) + t + str(statetoinsert)+t+'right'+t+str(DictToPrint[statetoinsert]['right'])+'\n'
                        f.write(writenext)
                        writenext = str(simnumber) + t + str(statetoinsert)+t+'left'+t+str(DictToPrint[statetoinsert]['left'])+'\n'
                        f.write(writenext)
                        writenext = str(simnumber) + t + str(statetoinsert)+t+'None'+t+str(DictToPrint[statetoinsert][None])+'\n'
                        f.write(writenext)
    return 1

def printLearningRate(f, deadline, simnumber,decisionnumber,netreward,posreward,negativereward,trnetreward,trposreward,trnegativereward,alpha,epsilon,gamma,randomsims):
    t = ','
    writenext = "Simulation Number,Number of Decisions,Deadline,Net Overall Reward,Overall Positive Reward,Overall Negative Reward,Current Sim Net Reward,Current Sim Positive Reward,Current Sim Negative Reward,Alpha,Epsilon,Gamma,Randomsims"+'\n'
    if simnumber == 1:
        print(writenext)
        f.write(writenext)

    writenext = str(simnumber)+t+str(decisionnumber) + t + str(deadline) + t + str(netreward) + t + str(posreward) + t + str(negativereward) + t + str(trnetreward) + t + str(trposreward) + t + str(trnegativereward)+ t + str(alpha)+ t + str(epsilon)+ t + str(gamma)+ t + str(randomsims)+'\n'
    print(writenext)
    f.write(writenext)
    return 1

def getdirection(DirectionNumber):
    direction = 'forward'
    if DirectionNumber == 1:
        direction = 'forward'
    elif DirectionNumber == 2:
        direction = 'right'
    elif DirectionNumber == 3:
        direction = 'left'
    else:
        direction = None
    return direction


def getdictinput(inputs,waypointaction):
    dictInput = ''
    dictInput = str(inputs['light']) +','+ str(inputs['oncoming']) + ',' + str(inputs['right']) + ',' + str(inputs['left'])+','+str(waypointaction)
    return dictInput


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track


    # Now simulate it
    sim = Simulator(e, update_delay=.1)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
