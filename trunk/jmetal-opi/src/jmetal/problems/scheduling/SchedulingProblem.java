package jmetal.problems.scheduling;

import jmetal.core.Problem;

public abstract class SchedulingProblem extends Problem {
	private static final long serialVersionUID = -7878751941443403073L;
	
	private int num_tasks;
	private int num_machines;
	
	public SchedulingProblem(int num_tasks, int num_machines) {
		this.num_tasks = num_tasks;
		this.num_machines = num_machines;
	}
	
	public int getNum_tasks() {
		return num_tasks;
	}
	
	public void setNum_tasks(int num_tasks) {
		this.num_tasks = num_tasks;
	}

	public int getNum_machines() {
		return num_machines;
	}

	public void setNum_machines(int num_machines) {
		this.num_machines = num_machines;
	}
}
