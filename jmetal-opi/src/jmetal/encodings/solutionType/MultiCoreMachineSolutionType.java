package jmetal.encodings.solutionType;

import jmetal.core.SolutionType;
import jmetal.core.Variable;
import jmetal.encodings.variable.MultiCoreMachine;
import jmetal.problems.scheduling.MultiCoreSchedulingProblem;
import jmetal.util.Configuration;
import jmetal.util.PseudoRandom;

public class MultiCoreMachineSolutionType extends SolutionType {

	public MultiCoreSchedulingProblem problem;
	
	public MultiCoreMachineSolutionType(MultiCoreSchedulingProblem problem) {
		super(problem);
		
		this.problem = problem;
	}

	@Override
	public Variable[] createVariables() throws ClassNotFoundException {
		Variable[] variables = new Variable[problem.NUM_MACHINES];

		for (int machine_id = 0; machine_id < problem.NUM_MACHINES; machine_id++) {
			variables[machine_id] = new MultiCoreMachine(this.problem, machine_id);
		}

		for (int task_id = 0; task_id < problem.NUM_TASKS; task_id++) {
			int bad_instance_count = 0;
			int task_cores = problem.TASK_CORES[task_id];
			
			int machine_id = PseudoRandom.randInt(0, problem.NUM_MACHINES-1);
			int machine_cores = problem.MACHINE_CORES[machine_id];
			while (machine_cores < task_cores) {
				machine_id = (machine_id + 1) % problem.NUM_MACHINES;
				machine_cores = problem.MACHINE_CORES[machine_id];
				
				if (bad_instance_count++ > (problem.NUM_MACHINES+1)) {
					Configuration.logger_.severe("ERROR! Invalid instance. Task too large for machines in scenario.");
					System.exit(-1);
				}
			}
			
			((MultiCoreMachine)variables[machine_id]).enqueue(task_id);
		}
		
		return variables;
	}

}
