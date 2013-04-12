package jmetal.problems.scheduling;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
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

	public double[] TASK_ARRIVAL;
	public int[] TASK_PRIORITY;
	public int[] TASK_CORES;
	public double[] TASK_COST;
	public int[] MACHINE_CORES;
	public double[] MACHINE_SSJ;
	public double[] MACHINE_EIDLE;
	public double[] MACHINE_EMAX;

	public MultiCoreSchedulingProblem(List<Double> task_arrival,
			List<Integer> task_priority, List<Integer> task_cores,
			List<Double> task_cost, List<Integer> machine_cores,
			List<Double> machine_ssj, List<Double> machine_eidle,
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

		TASK_ARRIVAL = new double[NUM_TASKS];
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
		MACHINE_SSJ = new double[NUM_MACHINES];
		MACHINE_EIDLE = new double[NUM_MACHINES];
		MACHINE_EMAX = new double[NUM_MACHINES];

		for (int i = 0; i < NUM_MACHINES; i++) {
			MACHINE_CORES[i] = machine_cores.get(i).intValue();
			MACHINE_SSJ[i] = machine_ssj.get(i).doubleValue();
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
		double energy = ((MultiCoreMachine) solution.getDecisionVariables()[0])
				.getEnergyConsumption();
		for (int m = 1; m < solution.numberOfVariables(); m++) {
			energy += ((MultiCoreMachine) solution.getDecisionVariables()[m])
					.getEnergyConsumption();

		}
		return energy;
	}

	private double evaluateWeightedCompletionTime(Solution solution) {
		double wct = ((MultiCoreMachine) solution.getDecisionVariables()[0])
				.getWeightedComputeTime();
		for (int m = 1; m < solution.numberOfVariables(); m++) {
			wct += ((MultiCoreMachine) solution.getDecisionVariables()[m])
					.getWeightedComputeTime();

		}
		return wct;
	}

	//private static final int scale_factor = 10000;
	private static final int scale_factor = 1;

	public static MultiCoreSchedulingProblem loadMultiCoreSchedulingProblem(
			String task_arrival_file, String task_priority_file,
			String task_cores_file, String task_cost_file, String machine_file) {

		System.out.println("task_arrival_file : " + task_arrival_file);
		System.out.println("task_priority_file: " + task_priority_file);
		System.out.println("task_cores_file   : " + task_cores_file);
		System.out.println("task_cost_file    : " + task_cost_file);
		System.out.println("machine_file      : " + machine_file);

		try {
			List<Integer> task_priority = new ArrayList<Integer>();
			List<Integer> task_cores = new ArrayList<Integer>();
			List<Double> task_cost = new ArrayList<Double>();
			List<Double> task_arrival = new ArrayList<Double>();

			List<String> input_files = new ArrayList<String>();
			input_files.add(task_priority_file);
			input_files.add(task_cores_file);

			List<List<Integer>> IntInputs = new ArrayList<List<Integer>>();
			IntInputs.add(task_priority);
			IntInputs.add(task_cores);

			for (int i = 0; i < IntInputs.size(); i++) {
				FileReader file = new FileReader(input_files.get(i));
				BufferedReader reader = new BufferedReader(file);

				String line;
				while ((line = reader.readLine()) != null) {
					if (!line.isEmpty()) {
						IntInputs.get(i).add(Integer.parseInt(line.trim()));
					}
				}

				reader.close();
				file.close();
			}

			{
				FileReader file = new FileReader(task_cost_file);
				BufferedReader reader = new BufferedReader(file);

				String line;
				while ((line = reader.readLine()) != null) {
					if (!line.isEmpty()) {
						task_cost.add(Double.parseDouble(line.trim())
								/ scale_factor);
					}
				}

				reader.close();
				file.close();
			}

			{
				FileReader file = new FileReader(task_arrival_file);
				BufferedReader reader = new BufferedReader(file);

				String[] data;
				String line;
				Integer count;
				while ((line = reader.readLine()) != null) {
					if (!line.isEmpty()) {
						data = line.split("\t");
						count = Integer.parseInt(data[1].trim());

						for (int j = 0; j < count; j++) {
							task_arrival.add(Double.parseDouble(data[0].trim())
									/ scale_factor);
						}
					}
				}

				reader.close();
				file.close();
			}

			List<Integer> machine_cores = new ArrayList<Integer>();
			List<Double> machine_ssj = new ArrayList<Double>();
			List<Double> machine_eidle = new ArrayList<Double>();
			List<Double> machine_emax = new ArrayList<Double>();

			{
				FileReader file = new FileReader(machine_file);
				BufferedReader reader = new BufferedReader(file);

				String line;
				String[] data;

				while ((line = reader.readLine()) != null) {
					if (!line.isEmpty()) {
						data = line.split("\t");
						assert (data.length == 4);

						machine_cores.add(Integer.parseInt(data[0]));
						machine_ssj.add(Double.parseDouble(data[1])
								/ scale_factor);
						machine_eidle.add(Double.parseDouble(data[2])
								/ scale_factor);
						machine_emax.add(Double.parseDouble(data[3])
								/ scale_factor);
					}
				}

				reader.close();
				file.close();
			}

			MultiCoreSchedulingProblem p = new MultiCoreSchedulingProblem(
					task_arrival, task_priority, task_cores, task_cost,
					machine_cores, machine_ssj, machine_eidle, machine_emax);

			return p;
		} catch (Exception e) {
			System.out.println(e.toString());
			e.printStackTrace();
			System.exit(-1);
			return null;
		}
	}

}
