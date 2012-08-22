package jmetal.util.listScheduling;

import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.util.JMException;

public class RandomMCT extends ListScheduler {
	
	public RandomMCT(Problem p) throws Exception {
		super(p);
		// TODO Auto-generated constructor stub
	}

	@Override
	public Solution compute() throws Exception {
		int startingPos = (int) Math.floor(Math.random() * p.taskCount_);
		int direction = (int) Math.round(Math.random());
		
		return compute(startingPos, direction);
	}
	
	protected Solution compute(int startingPos, int direction) throws ClassNotFoundException, JMException {
		Solution s = new Solution(this.p);
		//Variable[] variables = new Variable[this.p.taskCount_];
		
		int[] assign = new int[this.p.taskCount_];
		
		float[] machineComputeTime = new float[this.p.machineCount_];
		for (int i = 0; i < this.p.machineCount_; i++) {
			machineComputeTime[i] = 0;
		}
		
		int currentTask = 0;

		for (int t = 0; t < this.p.taskCount_; t++) {
			if (direction == 1) {
				currentTask = (startingPos + t) % this.p.taskCount_;
			} else {
				currentTask = startingPos - t;
				if (currentTask < 0) currentTask += this.p.taskCount_;
			}
			
			int bestMachine = 0;
			for (int m = 1; m < this.p.machineCount_; m++) {
				if (machineComputeTime[m] + this.p.workload_.ETC[m][currentTask]
						< machineComputeTime[bestMachine] + this.p.workload_.ETC[bestMachine][currentTask]) {
					
					bestMachine = m;
				}
			}
			
			assign[currentTask] = bestMachine;
			s.getDecisionVariables()[currentTask].setValue(bestMachine);
			machineComputeTime[bestMachine] += this.p.workload_.ETC[bestMachine][currentTask];
		}

		/*
		float makespan = 0;
		for (int i = 0; i < this.p.machineCount_; i++) {
			if (makespan < machineComputeTime[i]) {
				makespan = machineComputeTime[i];
			}
		}
		System.out.println("Solution makespan " + makespan);
		*/
		
		return s;
	}
}
