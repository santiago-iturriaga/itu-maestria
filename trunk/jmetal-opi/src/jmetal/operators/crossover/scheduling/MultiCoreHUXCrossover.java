package jmetal.operators.crossover.scheduling;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import jmetal.core.Solution;
import jmetal.encodings.solutionType.MultiCoreMachineSolutionType;
import jmetal.encodings.variable.Binary;
import jmetal.encodings.variable.MultiCoreMachine;
import jmetal.operators.crossover.Crossover;
import jmetal.problems.scheduling.MultiCoreSchedulingProblem;
import jmetal.util.Configuration;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;

public class MultiCoreHUXCrossover extends Crossover {
	private static final long serialVersionUID = -5539509679780123236L;

	/**
	 * Valid solution types to apply this operator
	 */
	private static List VALID_TYPES = Arrays
			.asList(MultiCoreMachineSolutionType.class);

	private Double probability_ = null;

	public MultiCoreHUXCrossover(HashMap<String, Object> parameters) {
		super(parameters);

		if (parameters.get("probability") != null)
			probability_ = (Double) parameters.get("probability");
	}

	/**
	 * Executes the operation
	 * 
	 * @param object
	 *            An object containing an array of two solutions
	 * @return An object containing the offSprings
	 */
	@Override
	public Object execute(Object object) throws JMException {
		Solution[] parents = (Solution[]) object;

		if (parents.length < 2) {

			Configuration.logger_
					.severe("HUXCrossover.execute: operator needs two parents");

			Class cls = java.lang.String.class;
			String name = cls.getName();
			throw new JMException("Exception in " + name + ".execute()");
		}

		if (!(VALID_TYPES.contains(parents[0].getType().getClass()) && VALID_TYPES
				.contains(parents[1].getType().getClass()))) {

			Configuration.logger_.severe("HUXCrossover.execute: the solutions "
					+ "are not of the right type. Inbound types are: "
					+ parents[0].getType() + " and " + parents[1].getType());

			Class cls = java.lang.String.class;
			String name = cls.getName();
			throw new JMException("Exception in " + name + ".execute()");
		} // if

		Solution[] offSpring = doCrossover(probability_, parents[0], parents[1]);

		for (int i = 0; i < offSpring.length; i++) {
			offSpring[i].setCrowdingDistance(0.0);
			offSpring[i].setRank(0);
		}

		return offSpring;
	} // execute

	/**
	 * Perform the crossover operation
	 * 
	 * @param probability
	 *            Crossover probability
	 * @param parent1
	 *            The first parent
	 * @param parent2
	 *            The second parent
	 * @return An array containing the two offsprings
	 * @throws JMException
	 */
	public Solution[] doCrossover(double probability, Solution parent1,
			Solution parent2) throws JMException {

		Solution[] offSpring = new Solution[2];
		offSpring[0] = new Solution(parent1);
		offSpring[1] = new Solution(parent2);

		try {
			if (PseudoRandom.randDouble() < probability) {
				for (int m_id = 0; m_id < parent1.getDecisionVariables().length; m_id++) {
					MultiCoreMachine p1 = (MultiCoreMachine) parent1
							.getDecisionVariables()[m_id];
					MultiCoreMachine p2 = (MultiCoreMachine) parent2
							.getDecisionVariables()[m_id];

					MultiCoreMachine o1 = (MultiCoreMachine) offSpring[0]
							.getDecisionVariables()[m_id];
					MultiCoreMachine o2 = (MultiCoreMachine) offSpring[1]
							.getDecisionVariables()[m_id];

					MultiCoreSchedulingProblem problem = p1.getProblem();

					int max_index = Math.min(o1.getMachine_tasks_count(),
							o2.getMachine_tasks_count());

					for (int t_index = 0; t_index < max_index; t_index++) {
						// if (p1.bits_.get(bit) != p2.bits_.get(bit)) {
						if (PseudoRandom.randDouble() < 0.5) {
							int p1_task_id = p1.getMachine_task(t_index);
							int p2_task_id = p2.getMachine_task(t_index);

							int o1_task_id = o1.getMachine_task(t_index);
							int o2_task_id = o2.getMachine_task(t_index);

							if (o2_task_id != p1_task_id) {
								int o2_task_machine_id = 0;
								MultiCoreMachine o2_task_machine = (MultiCoreMachine) offSpring[1]
										.getDecisionVariables()[o2_task_machine_id];

								while (!o2_task_machine
										.isTaskAssigned(p1_task_id)) {
									o2_task_machine_id++;
									o2_task_machine = (MultiCoreMachine) offSpring[1]
											.getDecisionVariables()[o2_task_machine_id];
								}

								int o2_task_machine_index = o2_task_machine
										.getTaskIndex(p1_task_id);

								if (o2_task_machine.getMachineId() == o2
										.getMachineId()) {
									o2.localSwapMachine_task(t_index,
											o2_task_machine_index);
								} else {
									if ((o2_task_machine.getMachineCores() >= problem.TASK_CORES[o2_task_id])
											&& (PseudoRandom.randDouble() < 0.5)) {
										o2_task_machine.swapMachine_task(
												o2_task_machine_index,
												o2_task_id);
										o2.swapMachine_task(t_index, p1_task_id);
									} else {
										o2_task_machine
												.removeMachine_task(o2_task_machine_index);
										o2.insertMachine_task(t_index,
												p1_task_id);

										max_index = Math.min(max_index,
												o2.getMachine_tasks_count());
									}
								}
							}

							if (o1_task_id != p2_task_id) {
								int o1_task_machine_id = 0;
								MultiCoreMachine o1_task_machine = (MultiCoreMachine) offSpring[0]
										.getDecisionVariables()[o1_task_machine_id];

								while (!o1_task_machine
										.isTaskAssigned(p2_task_id)) {
									o1_task_machine_id++;
									o1_task_machine = (MultiCoreMachine) offSpring[0]
											.getDecisionVariables()[o1_task_machine_id];
								}

								int o1_task_machine_index = o1_task_machine
										.getTaskIndex(p2_task_id);

								if (o1_task_machine.getMachineId() == o1
										.getMachineId()) {
									o1.localSwapMachine_task(t_index,
											o1_task_machine_index);
								} else {
									if ((o1_task_machine.getMachineCores() >= problem.TASK_CORES[o1_task_id])
											&& (PseudoRandom.randDouble() < 0.5)) {
										o1_task_machine.swapMachine_task(
												o1_task_machine_index,
												o1_task_id);
										o1.swapMachine_task(t_index, p2_task_id);
									} else {
										o1_task_machine
												.removeMachine_task(o1_task_machine_index);
										o1.insertMachine_task(t_index,
												p2_task_id);

										max_index = Math.min(
												o1.getMachine_tasks_count(),
												max_index);
									}
								}
							}
						}
					}

					o1.refresh();
					o2.refresh();
				}
			}
		} catch (ClassCastException e1) {
			Configuration.logger_
					.severe("HUXCrossover.doCrossover: Cannot perfom MultiCoreHUXCrossover");

			Class cls = java.lang.String.class;
			String name = cls.getName();
			throw new JMException("Exception in " + name + ".doCrossover()");
		}

		return offSpring;
	} // doCrossover
}
