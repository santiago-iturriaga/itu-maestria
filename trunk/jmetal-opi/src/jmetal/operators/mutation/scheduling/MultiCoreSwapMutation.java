package jmetal.operators.mutation.scheduling;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import jmetal.core.Solution;
import jmetal.encodings.solutionType.MultiCoreMachineSolutionType;
import jmetal.encodings.solutionType.PermutationSolutionType;
import jmetal.encodings.variable.Permutation;
import jmetal.operators.mutation.Mutation;
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
		int permutation[];
		int permutationLength;
		
		if (solution.getType().getClass() == MultiCoreMachineSolutionType.class) {
			/*
			permutationLength = ((Permutation) solution.getDecisionVariables()[0])
					.getLength();
			permutation = ((Permutation) solution.getDecisionVariables()[0]).vector_;

			if (PseudoRandom.randDouble() < probability) {
				int pos1;
				int pos2;

				pos1 = PseudoRandom.randInt(0, permutationLength - 1);
				pos2 = PseudoRandom.randInt(0, permutationLength - 1);

				while (pos1 == pos2) {
					if (pos1 == (permutationLength - 1))
						pos2 = PseudoRandom.randInt(0, permutationLength - 2);
					else
						pos2 = PseudoRandom
								.randInt(pos1, permutationLength - 1);
				} // while
					// swap
				int temp = permutation[pos1];
				permutation[pos1] = permutation[pos2];
				permutation[pos2] = temp;
			} // if
			*/
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
