package jmetal.encodings.variable;

import jmetal.core.Variable;
import jmetal.problems.scheduling.MultiCoreSchedulingProblem;

public class MultiCoreMachine extends Variable {
	private static final long serialVersionUID = -2428925055988408258L;
	
	private MultiCoreSchedulingProblem problem;
	private int machine_id;
	
	//private static Object shared_lock = new Object();	/* lock object to sync shared access */
	//private static int[] task_machine = null; 			/* executing machine assigned to each task */
	//private static int[][] task_cores = null; 			/* executing cores assigned to each task */
	
	private int machine_tasks_count; 	/* number of tasks assigned to the machine */
	private int[] machine_tasks;		/* tasks assigned to the machine */
	private double[] machine_core_ct;	/* local makespan of each core */
	private int[] machine_core_order;	/* sorted core (from min to max) */
	
	private double energy_consumption = 0.0;
	private double weighted_ct = 0;
	
	public MultiCoreMachine(MultiCoreSchedulingProblem problem, int machine_id) {	
		this.problem = problem;
		this.machine_id = machine_id;
		
		this.machine_tasks_count = 0;
		this.machine_tasks = new int[problem.NUM_TASKS];
		
		/*synchronized(shared_lock) {
			if (this.task_machine == null) {
				this.task_machine = new int[max_task_count];
				this.task_cores = new int[max_task_count][core_count];
				
				for (int i = 0; i < max_num_tasks; i++) {
					this.task_machine[i] = -1;
				}
			}
		}*/
		
		this.machine_core_ct = new double[problem.MACHINE_CORES[machine_id]];
		this.machine_core_order = new int[problem.MACHINE_CORES[machine_id]];
		for (int i = 0; i < problem.MACHINE_CORES[machine_id]; i++) {
			this.machine_core_ct[i] = 0;
			this.machine_core_order[i] = i;
		}
		
		this.energy_consumption = 0;
		this.weighted_ct = 0;
	}

	public MultiCoreMachine(MultiCoreMachine multiCoreMachine) {
		this.problem = multiCoreMachine.problem;
		this.machine_id = multiCoreMachine.machine_id;
			
		this.machine_tasks_count = multiCoreMachine.machine_tasks_count;
		
		this.machine_tasks = new int[multiCoreMachine.problem.NUM_TASKS];		
		for (int i = 0; i < this.machine_tasks_count; i++) {
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
	}
	
	public void enqueue(int task_id) {
		int task_cores = problem.TASK_CORES[task_id];
		int machine_cores = problem.MACHINE_CORES[machine_id];
		
		assert(task_cores <= machine_cores);
		
		machine_tasks[machine_tasks_count] = task_id;
		machine_tasks_count++;
		
		int assigned_worst_core_id = this.machine_core_order[task_cores-1];
		double assigned_worst_core_ct = this.machine_core_ct[assigned_worst_core_id] + problem.TASK_COST[task_id];
		
		for (int i = 0; i < task_cores; i++) {
			this.machine_core_ct[this.machine_core_order[i]] = assigned_worst_core_ct;
		}
		
		int aux_order;
		for (int i = task_cores-1; i < machine_cores-1; i++) {
			if (this.machine_core_ct[this.machine_core_order[i]] > this.machine_core_ct[this.machine_core_order[i+1]]) {
				aux_order = this.machine_core_order[i+1];
				this.machine_core_order[i+1] = this.machine_core_order[i-task_cores+1];
				this.machine_core_order[i-task_cores+1] = aux_order;
			} else {
				break;
			}
		}
	}
	
	/*
	public double getCore_CT(int core_idx) {
		assert(core_idx < this.num_cores);
		
		return this.core_ct[core_idx];
	}
	
	public int getTask(int core_idx, int task_pos) {
		assert(core_idx < this.num_cores);
		assert(task_pos < this.max_num_tasks);
		
		return core_tasks[core_idx][task_pos];
	}
	
	public int[] getCore_tasks_count() {
		return core_tasks_count;
	}

	public void setCore_tasks_count(int[] core_tasks_count) {
		this.core_tasks_count = core_tasks_count;
	}

	public int getTasks_count() {
		return tasks_count;
	}

	public void setTasks_count(int tasks_count) {
		this.tasks_count = tasks_count;
	}
	*/
	
	/*
	public void setTask(int core_idx, int task_idx, int task_pos) {
		assert(core_idx < this.core_count);
		assert(task_pos < this.max_task_count);
		
		assigned_tasks[core_idx][task_pos] = task_idx;
	}
	
	public void appendTask(int )
	*/
	
	@Override
	public String toString() {
		String output = "[machine " + this.machine_id + "] ";
		for (int i = 0; i < this.machine_tasks_count; i++) {
			output += this.machine_tasks[i] + " ";
		}
		return output;
	}
	
	@Override
	public Variable deepCopy() {
		return new MultiCoreMachine(this);
	}
}
