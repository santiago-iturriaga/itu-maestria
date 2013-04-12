package jmetal.operators.mutation.scheduling;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import jmetal.core.Solution;
import jmetal.encodings.solutionType.MultiCoreMachineSolutionType;
import jmetal.encodings.solutionType.PermutationSolutionType;
import jmetal.encodings.variable.MultiCoreMachine;
import jmetal.encodings.variable.Permutation;
import jmetal.operators.mutation.Mutation;
import jmetal.problems.scheduling.MultiCoreSchedulingProblem;
import jmetal.util.Configuration;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;

public class MultiCoreSwapMutation extends Mutation {

	private static final long serialVersionUID = 1393282187956101782L;

	/**
	 * Valid solution types to apply this operator
	 */
	private static List VALID_TYPES = Arrays
			.asList(MultiCoreMachineSolutionType.class);

	private Double mutationProbability_ = null;

	public MultiCoreSwapMutation(HashMap<String, Object> parameters) {
		super(parameters);

		if (parameters.get("probability") != null)
			mutationProbability_ = (Double) parameters.get("probability");
	}

	/**
	 * Executes the operation
	 * 
	 * @param object
	 *            An object containing the solution to mutate
	 * @return an object containing the mutated solution
	 * @throws JMException
	 */
	@Override
	public Object execute(Object object) throws JMException {
		Solution solution = (Solution) object;

		if (!VALID_TYPES.contains(solution.getType().getClass())) {
			Configuration.logger_.severe("SwapMutation.execute: the solution "
					+ "is not of the right type. Inbound type is "
					+ solution.getType());

			Class cls = java.lang.String.class;
			String name = cls.getName();
			throw new JMException("Exception in " + name + ".execute()");
		} // if

		this.doMutation(mutationProbability_, solution);
		return solution;
	} // execute

	/**
	 * Performs the operation
	 * 
	 * @param probability
	 *            Mutation probability
	 * @param solution
	 *            The solution to mutate
	 * @throws JMException
	 */
	public void doMutation(double probability, Solution solution)
			throws JMException {

		boolean[] modified_machines = new boolean[solution.numberOfVariables()];
		for (int i = 0; i < solution.numberOfVariables(); i++)
			modified_machines[i] = false;

		if (solution.getType().getClass() == MultiCoreMachineSolutionType.class) {
			for (int m_orig_id = 0; m_orig_id < solution.numberOfVariables(); m_orig_id++) {
				MultiCoreMachine m_orig = ((MultiCoreMachine) (solution
						.getDecisionVariables()[m_orig_id]));
				int m_tasks = m_orig.getMachine_tasks_count();

				for (int t_orig_index = 0; t_orig_index < m_tasks; t_orig_index++) {
					if (PseudoRandom.randDouble() < probability) {
						MultiCoreSchedulingProblem problem = m_orig
								.getProblem();

						int t_orig_id = m_orig.getMachine_task(t_orig_index);
						int t_orig_cores = problem.TASK_CORES[t_orig_id];

						int action_type = 0; // PseudoRandom.randInt(0, 2);
						assert (action_type < 3);

						if (action_type == 0) {
							// Random swap
							int m_dest_id;
							m_dest_id = PseudoRandom.randInt(0,
									solution.numberOfVariables());

							while (problem.MACHINE_CORES[m_dest_id] < t_orig_cores) {
								m_dest_id = (m_dest_id + 1)
										% solution.numberOfVariables();
							}

							MultiCoreMachine m_dest;
							m_dest = (MultiCoreMachine) (solution
									.getDecisionVariables()[m_dest_id]);

							if (m_dest.getMachine_tasks_count() > 0) {
								int t_dest_index;
								t_dest_index = PseudoRandom.randInt(0,
										m_dest.getMachine_tasks_count());

								int t_dest_id = m_dest
										.getMachine_task(t_dest_index);
								int t_dest_cores = problem.TASK_CORES[t_dest_id];

								if (problem.MACHINE_CORES[m_orig_id] >= t_dest_cores) {
									if (!((m_dest_id == m_orig_id) && (t_dest_index == t_orig_index))) {
										int aux_orig = m_orig
												.getMachine_task(t_orig_index);

										if (m_orig.getMachineId() != m_dest
												.getMachineId()) {
											m_orig.swapMachine_task(
													t_orig_index, t_dest_id);
											m_dest.swapMachine_task(
													t_dest_index, aux_orig);
										} else {
											m_orig.localSwapMachine_task(
													t_orig_index, t_dest_index);
										}

										modified_machines[m_orig_id] = true;
										modified_machines[m_dest_id] = true;
									}
								} else {
									m_dest.enqueue(t_orig_id);
									m_orig.removeMachine_task(t_orig_index);
									modified_machines[m_orig_id] = true;
									
									m_tasks = m_orig.getMachine_tasks_count();
									t_orig_index--;
								}
							}
						} else if (action_type == 1) {
							// Random move
							int m_dest_id;
							m_dest_id = PseudoRandom.randInt(0,
									solution.numberOfVariables());

							while (problem.MACHINE_CORES[m_dest_id] < t_orig_cores) {
								m_dest_id = (m_dest_id + 1)
										% solution.numberOfVariables();
							}

							MultiCoreMachine m_dest;
							m_dest = (MultiCoreMachine) (solution
									.getDecisionVariables()[m_dest_id]);

							if (m_dest_id != m_orig_id) {
								m_dest.enqueue(t_orig_id);
								m_orig.removeMachine_task(t_orig_index);

								modified_machines[m_orig_id] = true;

								m_tasks = m_orig.getMachine_tasks_count();
								t_orig_index--;

							} else {
								int t_dest_index;
								t_dest_index = PseudoRandom.randInt(0, m_tasks);

								if (t_dest_index != t_orig_index) {
									m_orig.localSwapMachine_task(t_orig_index,
											t_dest_index);
									modified_machines[m_orig_id] = true;
								}
							}
						} else if (action_type == 2) {
							// Move forward or backward

							int t_dest_index;
							t_dest_index = PseudoRandom.randInt(0, m_tasks);

							if (t_dest_index != t_orig_index) {
								m_orig.localSwapMachine_task(t_orig_index,
										t_dest_index);
								modified_machines[m_orig_id] = true;
							}
						}
					}
				}
			}

			for (int i = 0; i < solution.numberOfVariables(); i++) {
				if (modified_machines[i]) {
					((MultiCoreMachine) (solution.getDecisionVariables()[i]))
							.refresh();
				}
			}
		} // if
		else {
			Configuration.logger_
					.severe("SwapMutation.doMutation: invalid type "
							+ solution.getDecisionVariables()[0]
									.getVariableType());

			Class cls = java.lang.String.class;
			String name = cls.getName();
			throw new JMException("Exception in " + name + ".doMutation()");
		}
	} // doMutation
}
