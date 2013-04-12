package jmetal.encodings.variable;

import java.util.*;
import jmetal.core.Variable;
import jmetal.problems.scheduling.MultiCoreSchedulingProblem;

public class MultiCoreMachine extends Variable {
	private static final long serialVersionUID = -2428925055988408258L;

	private MultiCoreSchedulingProblem problem;
	private int machine_id;
	private int machine_cores;
	private double consumption_per_core;
	private double ssj_per_core;

	private int machine_tasks_count; /* number of tasks assigned to the machine */
	private int[] machine_tasks; /* tasks assigned to the machine */
	private double[] machine_core_ct; /* local makespan of each core */
	private int[] machine_core_order; /* sorted core (from min to max) */

	private double energy_consumption = 0.0;
	private double total_executing_time = 0.0;
	private double weighted_ct = 0;

	private HashSet<Integer> assignedTasks;

	public MultiCoreMachine(MultiCoreSchedulingProblem problem, int machine_id) {
		this.problem = problem;
		this.machine_id = machine_id;

		this.machine_cores = problem.MACHINE_CORES[machine_id];
		this.consumption_per_core = (problem.MACHINE_EMAX[machine_id] - problem.MACHINE_EIDLE[machine_id])
				/ problem.MACHINE_CORES[machine_id];
		this.ssj_per_core = (problem.MACHINE_SSJ[machine_id] / problem.MACHINE_CORES[machine_id]);

		this.machine_tasks_count = 0;
		this.machine_tasks = new int[problem.NUM_TASKS];

		this.machine_core_ct = new double[problem.MACHINE_CORES[machine_id]];
		this.machine_core_order = new int[problem.MACHINE_CORES[machine_id]];
		for (int i = 0; i < problem.MACHINE_CORES[machine_id]; i++) {
			this.machine_core_ct[i] = 0;
			this.machine_core_order[i] = i;
		}

		this.energy_consumption = 0;
		this.weighted_ct = 0;
		this.total_executing_time = 0;
		this.assignedTasks = new HashSet<Integer>();
	}

	public MultiCoreMachine(MultiCoreMachine multiCoreMachine) {
		this.problem = multiCoreMachine.problem;
		this.machine_id = multiCoreMachine.machine_id;

		this.machine_cores = multiCoreMachine.machine_cores;
		this.consumption_per_core = multiCoreMachine.consumption_per_core;
		this.ssj_per_core = multiCoreMachine.ssj_per_core;

		this.machine_tasks_count = multiCoreMachine.getMachine_tasks_count();
		this.machine_tasks = new int[multiCoreMachine.problem.NUM_TASKS];
		for (int i = 0; i < this.getMachine_tasks_count(); i++) {
			this.machine_tasks[i] = multiCoreMachine.machine_tasks[i];
		}

		this.machine_core_ct = new double[problem.MACHINE_CORES[machine_id]];
		this.machine_core_order = new int[problem.MACHINE_CORES[machine_id]];
		for (int i = 0; i < problem.MACHINE_CORES[machine_id]; i++) {
			this.machine_core_ct[i] = multiCoreMachine.machine_core_ct[i];
			this.machine_core_order[i] = multiCoreMachine.machine_core_order[i];
		}

		this.energy_consumption = multiCoreMachine.energy_consumption;
		this.weighted_ct = multiCoreMachine.weighted_ct;
		this.total_executing_time = multiCoreMachine.total_executing_time;
		this.assignedTasks = new HashSet<Integer>(
				multiCoreMachine.assignedTasks);
	}

	public MultiCoreSchedulingProblem getProblem() {
		return (MultiCoreSchedulingProblem) this.problem;
	}

	public void refresh() {
		for (int i = 0; i < problem.MACHINE_CORES[machine_id]; i++) {
			this.machine_core_ct[i] = 0;
			this.machine_core_order[i] = i;
		}

		this.energy_consumption = 0;
		this.weighted_ct = 0;
		this.total_executing_time = 0;

		for (int i = 0; i < getMachine_tasks_count(); i++) {
			addTaskComputation(machine_tasks[i]);
		}
	}

	public void enqueue(int task_id) {
		this.machine_tasks[getMachine_tasks_count()] = task_id;
		this.assignedTasks.add(task_id);
		this.machine_tasks_count++;

		addTaskComputation(task_id);
	}

	private void addTaskComputation(int task_id) {
		int task_cores = problem.TASK_CORES[task_id];
		assert (task_cores <= machine_cores);

		int assigned_worst_core_id = this.machine_core_order[task_cores - 1];

		/* Calculo el starting time */
		double starting_time = this.machine_core_ct[assigned_worst_core_id];
		if (starting_time <= problem.TASK_ARRIVAL[task_id])
			starting_time = problem.TASK_ARRIVAL[task_id];

		/* Actualizo el weighted compute time */
		// this.weighted_ct += starting_time * problem.TASK_PRIORITY[task_id];
		this.weighted_ct += (starting_time - problem.TASK_ARRIVAL[task_id])
				* problem.TASK_PRIORITY[task_id];

		double task_executing_time = (problem.TASK_COST[task_id] / this.ssj_per_core)
				* task_cores;
		this.total_executing_time += task_executing_time;

		/* Actualizo la energía consumida por la máquina */
		this.energy_consumption += task_executing_time
				* this.consumption_per_core;

		/* Calculo el ending time */
		double assigned_worst_core_ct = starting_time
				+ (problem.TASK_COST[task_id] / this.ssj_per_core);

		for (int i = 0; i < task_cores; i++) {
			this.machine_core_ct[this.machine_core_order[i]] = assigned_worst_core_ct;
		}

		/* Re-ordeno los cores por compute time */
		int aux_order;
		for (int i = task_cores - 1; i < this.machine_cores - 1; i++) {
			if (this.machine_core_ct[this.machine_core_order[i]] > this.machine_core_ct[this.machine_core_order[i + 1]]) {
				aux_order = this.machine_core_order[i + 1];
				this.machine_core_order[i + 1] = this.machine_core_order[i
						- task_cores + 1];
				this.machine_core_order[i - task_cores + 1] = aux_order;
			} else {
				break;
			}
		}
	}

	public int getMachineId() {
		return this.machine_id;
	}
	
	public int getMachineCores() {
		return this.machine_cores;
	}

	public double getTotalComputeTime() {
		return machine_core_ct[machine_core_order[this.machine_cores - 1]];
	}

	public double getWeightedComputeTime() {
		return this.weighted_ct;
	}

	public double getExecutingTime() {
		return this.total_executing_time;
	}

	public double getEnergyConsumption() {
		return this.energy_consumption;
	}

	public int getMachine_task(int queue_index) {
		assert (queue_index < this.machine_tasks_count);
		return this.machine_tasks[queue_index];
	}

	public int getMachine_tasks_count() {
		return machine_tasks_count;
	}

	public double[] getMachine_core_ct() {
		return this.machine_core_ct;
	}

	public double getConsumption_per_core() {
		return this.consumption_per_core;
	}

	public double getSsj_per_core() {
		return this.ssj_per_core;
	}

	public boolean isTaskAssigned(int task_id) {
		return this.assignedTasks.contains(task_id);
	}

	public int getTaskIndex(int task_id) {
		assert (isTaskAssigned(task_id));
		int index = 0;
		while (this.machine_tasks[index] != task_id)
			index++;
		return index;
	}

	public void insertMachine_task(int queue_index, int task_id) {
		assert (queue_index < this.machine_tasks_count);
		assert (problem.TASK_CORES[task_id] <= machine_cores);

		for (int i = this.machine_tasks_count - 1; i >= queue_index; i--) {
			this.machine_tasks[i + 1] = this.machine_tasks[i];
		}

		this.machine_tasks[queue_index] = task_id;
		this.machine_tasks_count++;
		this.assignedTasks.add(task_id);
		
		assert (check_integrity());
	}

	public void removeMachine_task(int queue_index) {
		assert (queue_index < this.machine_tasks_count);

		this.assignedTasks.remove(this.machine_tasks[queue_index]);

		for (int i = queue_index; i < this.machine_tasks_count - 1; i++) {
			this.machine_tasks[i] = this.machine_tasks[i + 1];
		}

		this.machine_tasks_count--;
	}

	public void localSwapMachine_task(int queue_index_1, int queue_index_2) {
		assert (queue_index_1 < this.machine_tasks_count);
		assert (queue_index_2 < this.machine_tasks_count);

		int aux = this.machine_tasks[queue_index_1];
		this.machine_tasks[queue_index_1] = this.machine_tasks[queue_index_2];
		this.machine_tasks[queue_index_2] = aux;
	}

	public void swapMachine_task(int queue_index, int task_id) {
		assert (queue_index < this.machine_tasks_count);
		assert (problem.TASK_CORES[task_id] <= machine_cores);

		int old_task_id = this.machine_tasks[queue_index];

		this.assignedTasks.remove(old_task_id);
		this.assignedTasks.add(task_id);

		this.machine_tasks[queue_index] = task_id;

		assert (check_integrity());
	}

	private boolean check_integrity() {
		for (int i = 0; i < this.machine_tasks_count; i++) {
			if (!this.assignedTasks.contains(this.machine_tasks[i])) {
				return false;
			}
		}

		return true;
	}

	@Override
	public String toString() {
		String output = "[machine " + this.machine_id + "] ";
		for (int i = 0; i < this.getMachine_tasks_count(); i++) {
			output += this.machine_tasks[i] + " ";
		}
		return output;
	}

	@Override
	public Variable deepCopy() {
		return new MultiCoreMachine(this);
	}
}
