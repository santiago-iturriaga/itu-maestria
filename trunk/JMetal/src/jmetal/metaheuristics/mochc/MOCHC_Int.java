package jmetal.metaheuristics.mochc;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.encodings.variable.Int;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;
import jmetal.util.archive.CrowdingArchive;
import jmetal.util.comparators.CrowdingComparator;

import java.util.*;

/**
 * 
 * Class implementing the CHC algorithm.
 */
public class MOCHC_Int extends Algorithm {
	/**
	 * Constructor Creates a new instance of MOCHC
	 */
	public MOCHC_Int(Problem problem) {
		super(problem);
	}

	/**
	 * Compares two solutionSets to determine if both are equals
	 * 
	 * @param solutionSet
	 *            A <code>SolutionSet</code>
	 * @param newSolutionSet
	 *            A <code>SolutionSet</code>
	 * @return true if both are cotains the same solutions, false in other case
	 */
	public boolean equals(SolutionSet solutionSet, SolutionSet newSolutionSet) {
		boolean found;
		for (int i = 0; i < solutionSet.size(); i++) {

			int j = 0;
			found = false;
			while (j < newSolutionSet.size()) {

				if (solutionSet.get(i).equals(newSolutionSet.get(j))) {
					found = true;
				}
				j++;
			}
			if (!found) {
				return false;
			}
		}
		return true;
	} // equals

	public int distance(Solution solutionOne, Solution solutionTwo) {
		int distance = 0;
		for (int i = 0; i < problem_.getNumberOfVariables(); i++) {
			Int v1 = ((Int) solutionOne.getDecisionVariables()[i]);
			Int v2 = ((Int) solutionTwo.getDecisionVariables()[i]);

			if (v1.getValue() != v2.getValue()) {
				distance++;
			}
		}

		return distance;
	} // hammingDistance

	/**
	 * Runs of the MOCHC algorithm.
	 * 
	 * @return a <code>SolutionSet</code> that is a set of non dominated
	 *         solutions as a result of the algorithm execution
	 * @throws ClassNotFoundException 
	 */
	public SolutionSet execute() throws JMException, ClassNotFoundException {
		int iterations;
		int populationSize;
		int convergenceValue;
		int maxEvaluations;
		int minimumDistance;
		int evaluations;

		Comparator crowdingComparator = new CrowdingComparator();

		Operator crossover;
		Operator parentSelection;
		Operator newGenerationSelection;
		Operator cataclysmicMutation;

		double preservedPopulation;
		double initialConvergenceCount;
		boolean condition = false;
		SolutionSet solutionSet, offspringPopulation, newPopulation;

		// Read parameters
		initialConvergenceCount = ((Double) getInputParameter("initialConvergenceCount"))
				.doubleValue();
		preservedPopulation = ((Double) getInputParameter("preservedPopulation"))
				.doubleValue();
		convergenceValue = ((Integer) getInputParameter("convergenceValue"))
				.intValue();
		populationSize = ((Integer) getInputParameter("populationSize"))
				.intValue();
		maxEvaluations = ((Integer) getInputParameter("maxEvaluations"))
				.intValue();

		// Read operators
		crossover = (Operator) getOperator("crossover");
		cataclysmicMutation = (Operator) getOperator("cataclysmicMutation");
		parentSelection = (Operator) getOperator("parentSelection");
		newGenerationSelection = (Operator) getOperator("newGenerationSelection");

		iterations = 0;
		evaluations = 0;

		// Calculate the maximum problem sizes
		int size = problem_.getNumberOfVariables();
		minimumDistance = (int) Math.floor(initialConvergenceCount * size);

		solutionSet = new SolutionSet(populationSize);
		List<Solution> initialPopulation = ((List<Solution>) getInputParameter("initialPopulation"));

		if (initialPopulation != null) {
			for (int i = 0; (i < populationSize)
					&& (i < initialPopulation.size()); i++) {
				Solution solution;
				solution = new Solution(initialPopulation.get(i));
				problem_.evaluate(solution);
				problem_.evaluateConstraints(solution);
				evaluations++;
				solutionSet.add(solution);
			}
		}

		for (int i = solutionSet.size(); i < populationSize; i++) {
			Solution solution;
			solution = new Solution(problem_);
			problem_.evaluate(solution);
			problem_.evaluateConstraints(solution);
			evaluations++;
			solutionSet.add(solution);
		} // for

		while (!condition) {
			offspringPopulation = new SolutionSet(populationSize);
			for (int i = 0; i < solutionSet.size() / 2; i++) {
				Solution[] parents = (Solution[]) parentSelection
						.execute(solutionSet);

				// Equality condition between solutions
				if (distance(parents[0], parents[1]) >= (minimumDistance)) {
					Solution[] offspring = (Solution[]) crossover
							.execute(parents);
					problem_.evaluate(offspring[0]);
					problem_.evaluateConstraints(offspring[0]);
					problem_.evaluate(offspring[1]);
					problem_.evaluateConstraints(offspring[1]);
					evaluations += 2;
					offspringPopulation.add(offspring[0]);
					offspringPopulation.add(offspring[1]);
				}
			}
			SolutionSet union = solutionSet.union(offspringPopulation);
			newGenerationSelection.setParameter("populationSize",
					populationSize);
			newPopulation = (SolutionSet) newGenerationSelection.execute(union);

			if (equals(solutionSet, newPopulation)) {
				minimumDistance--;
			}
			if (minimumDistance <= -convergenceValue) {

				minimumDistance = (int) (1.0 / size * (1 - 1.0 / size) * size);
				// minimumDistance = (int) (0.35 * (1 - 0.35) * size);

				int preserve = (int) Math.floor(preservedPopulation
						* populationSize);
				newPopulation = new SolutionSet(populationSize);
				solutionSet.sort(crowdingComparator);
				for (int i = 0; i < preserve; i++) {
					newPopulation.add(new Solution(solutionSet.get(i)));
				}
				for (int i = preserve; i < populationSize; i++) {
					Solution solution = new Solution(solutionSet.get(i));
					cataclysmicMutation.execute(solution);
					problem_.evaluate(solution);
					problem_.evaluateConstraints(solution);
					newPopulation.add(solution);
				}
			}

			iterations++;

			solutionSet = newPopulation;
			if (evaluations >= maxEvaluations) {
				condition = true;
			}
		}

		CrowdingArchive archive;
		archive = new CrowdingArchive(populationSize,
				problem_.getNumberOfObjectives());
		for (int i = 0; i < solutionSet.size(); i++) {
			archive.add(solutionSet.get(i));
		}

		return archive;
	} // execute
} // MOCHC