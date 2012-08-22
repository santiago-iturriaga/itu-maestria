package jmetal.experiments.settings;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.experiments.Settings;
import jmetal.metaheuristics.moead.MOEAD;
import jmetal.metaheuristics.moead.pMOEAD;
import jmetal.metaheuristics.nsgaII.NSGAII;
import jmetal.operators.crossover.Crossover;
import jmetal.operators.crossover.CrossoverFactory;
import jmetal.operators.mutation.Mutation;
import jmetal.operators.mutation.MutationFactory;
import jmetal.operators.selection.Selection;
import jmetal.operators.selection.SelectionFactory;
import jmetal.problems.ProblemFactory;
import jmetal.problems.scheduling.MEProblem;
import jmetal.problems.scheduling.MEProblem.Scenario;
import jmetal.problems.scheduling.MEProblem.Workload;
import jmetal.qualityIndicator.QualityIndicator;
import jmetal.util.JMException;
import jmetal.util.listScheduling.RandomMCT;

public class MEScheduling_MOEAD_Settings extends Settings {
	public double CR_;
	public double F_;
	public int populationSize_;
	public int maxEvaluations_;

	public double mutationProbability_;
	public double distributionIndexForMutation_;

	public String dataDirectory_;

	public int numberOfThreads; // Parameter used by the pMOEAD version
	public String moeadVersion;

	public MEScheduling_MOEAD_Settings(String problemName)
			throws ClassNotFoundException, IOException {
		super(problemName);

		String[] problemInfo = problemName.split(" ");
		assert (problemInfo.length == 4);
		String[] dimension = problemInfo[1].split("x");

		int taskCount = Integer.parseInt(dimension[0]);
		int machineCount = Integer.parseInt(dimension[1]);
		String scenarioPath = "/home/santiago/Scheduling/Energy-Makespan/instances.ruso/"
				+ problemInfo[1] + "/" + problemInfo[2];
		String workloadPath = "/home/santiago/Scheduling/Energy-Makespan/instances.ruso/"
				+ problemInfo[1] + "/" + problemInfo[3];

		problem_ = new MEProblem(taskCount, machineCount, scenarioPath,
				workloadPath);

		// Default settings
		CR_ = 1.0;
		F_ = 0.5;
		populationSize_ = 300;
		maxEvaluations_ = 150000;

		mutationProbability_ = 1.0 / problem_.getNumberOfVariables();
		distributionIndexForMutation_ = 20;

		// Directory with the files containing the weight vectors used in
		// Q. Zhang, W. Liu, and H Li, The Performance of a New Version of
		// MOEA/D
		// on CEC09 Unconstrained MOP Test Instances Working Report CES-491,
		// School
		// of CS & EE, University of Essex, 02/2009.
		// http://dces.essex.ac.uk/staff/qzhang/MOEAcompetition/CEC09final/code/ZhangMOEADcode/moead0305.rar
		dataDirectory_ = "/Users/antonio/Softw/pruebas/data/MOEAD_parameters/Weight";

		numberOfThreads = 2; // Parameter used by the pMOEAD version
		moeadVersion = "MOEAD"; // or "pMOEAD"
	}

	/**
	 * Configure the algorithm with the specified parameter settings
	 * 
	 * @return an algorithm object
	 * @throws jmetal.util.JMException
	 */
	public Algorithm configure() throws JMException {
		Algorithm algorithm;
		Operator crossover;
		Operator mutation;

		QualityIndicator indicators;

		HashMap parameters; // Operator parameters

		// Creating the problem
		if (moeadVersion.compareTo("MOEAD") == 0)
			algorithm = new MOEAD(problem_);
		else { // pMOEAD
			algorithm = new pMOEAD(problem_);
			algorithm.setInputParameter("numberOfThreads", numberOfThreads);
		} // else

		// Algorithm parameters
		algorithm.setInputParameter("populationSize", populationSize_);
		algorithm.setInputParameter("maxEvaluations", maxEvaluations_);
		algorithm.setInputParameter("dataDirectory", dataDirectory_);

		try {
			RandomMCT initMethod = new RandomMCT(problem_);
			Solution initSolution;

			List<Solution> population = new LinkedList<Solution>();
			for (int i = 0; i < populationSize_; i++) {
				initSolution = initMethod.compute();
				population.add(initSolution);
			}

			algorithm.setInputParameter("initialPopulation", population);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		// Crossover operator
		parameters = new HashMap();
		parameters.put("CR", CR_);
		parameters.put("F", F_);
		crossover = CrossoverFactory.getCrossoverOperator(
				"DifferentialEvolutionCrossover", parameters);

		// Mutation operator
		parameters = new HashMap();
		parameters.put("probability", 1.0 / problem_.getNumberOfVariables());
		parameters.put("distributionIndex", distributionIndexForMutation_);
		mutation = MutationFactory.getMutationOperator("PolynomialMutation",
				parameters);

		algorithm.addOperator("crossover", crossover);
		algorithm.addOperator("mutation", mutation);

		// Creating the indicator object
		if ((paretoFrontFile_ != null) && (!paretoFrontFile_.equals(""))) {
			indicators = new QualityIndicator(problem_, paretoFrontFile_);
			algorithm.setInputParameter("indicators", indicators);
		} // if

		return algorithm;
	} // configure
}
