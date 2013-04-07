package jmetal.problems.scheduling;

import java.util.List;

import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.encodings.solutionType.BinaryRealSolutionType;
import jmetal.encodings.solutionType.MultiCoreMachineSolutionType;
import jmetal.encodings.variable.MultiCoreMachine;
import jmetal.util.JMException;

public class MultiCoreSchedulingProblem extends Problem {
	private static final long serialVersionUID = -781343273701991756L;

	public static final int WEIGHTED_COMPLETION_TIME_OBJNUM = 0;
	public static final int ENERGY_CONSUMPTION_OBJNUM = 1;

	public int NUM_TASKS;
	public int NUM_MACHINES;

	public int[] TASK_ARRIVAL;
	public int[] TASK_PRIORITY;
	public int[] TASK_CORES;
	public double[] TASK_COST;
	public int[] MACHINE_CORES;
	public int[] MACHINE_SSJ;
	public double[] MACHINE_EIDLE;
	public double[] MACHINE_EMAX;

	public MultiCoreSchedulingProblem(List<Integer> task_arrival,
			List<Integer> task_priority, List<Integer> task_cores,
			List<Double> task_cost, List<Integer> machine_cores,
			List<Integer> machine_ssj, List<Double> machine_eidle,
			List<Double> machine_emax) {

		NUM_TASKS = task_arrival.size();
		NUM_MACHINES = machine_cores.size();

		this.problemName_ = "MultiCoreScheduling";
		this.solutionType_ = new MultiCoreMachineSolutionType(this);
		this.numberOfVariables_ = NUM_MACHINES;
		this.numberOfObjectives_ = 2;
		this.numberOfConstraints_ = 0;

		assert (NUM_TASKS == task_arrival.size());
		assert (NUM_TASKS == task_priority.size());
		assert (NUM_TASKS == task_cores.size());
		assert (NUM_TASKS == task_cost.size());

		TASK_ARRIVAL = new int[NUM_TASKS];
		TASK_PRIORITY = new int[NUM_TASKS];
		TASK_CORES = new int[NUM_TASKS];
		TASK_COST = new double[NUM_TASKS];

		for (int i = 0; i < NUM_TASKS; i++) {
			TASK_ARRIVAL[i] = task_arrival.get(i).intValue();
			TASK_PRIORITY[i] = task_priority.get(i).intValue();
			TASK_CORES[i] = task_cores.get(i).intValue();
			TASK_COST[i] = task_cost.get(i).doubleValue();
		}

		assert (NUM_MACHINES == machine_cores.size());
		assert (NUM_MACHINES == machine_ssj.size());
		assert (NUM_MACHINES == machine_eidle.size());
		assert (NUM_MACHINES == machine_emax.size());

		MACHINE_CORES = new int[NUM_MACHINES];
		MACHINE_SSJ = new int[NUM_MACHINES];
		MACHINE_EIDLE = new double[NUM_MACHINES];
		MACHINE_EMAX = new double[NUM_MACHINES];

		for (int i = 0; i < NUM_MACHINES; i++) {
			MACHINE_CORES[i] = machine_cores.get(i).intValue();
			MACHINE_SSJ[i] = machine_ssj.get(i).intValue();
			MACHINE_EIDLE[i] = machine_eidle.get(i).doubleValue();
			MACHINE_EMAX[i] = machine_emax.get(i).doubleValue();
		}
	}

	@Override
	public void evaluate(Solution solution) throws JMException {
		double energy = evaluateEnergyConsumption(solution);
		double completion = evaluateWeightedCompletionTime(solution);

		solution.setObjective(WEIGHTED_COMPLETION_TIME_OBJNUM, completion);
		solution.setObjective(ENERGY_CONSUMPTION_OBJNUM, energy);
	}

	private double evaluateEnergyConsumption(Solution solution) {
		// (MultiCoreMachineSolutionType)

		for (int m = 0; m < solution.numberOfVariables(); m++) {
			// int cant_tareas =
			// ((MultiCoreMachine)solution.getDecisionVariables()[m]).getTasks_count();

		}

		return 0;
	}

	private double evaluateWeightedCompletionTime(Solution solution) {
		return 0;
	}

}
