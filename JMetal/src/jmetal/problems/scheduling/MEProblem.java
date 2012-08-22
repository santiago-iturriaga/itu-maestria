package jmetal.problems.scheduling;

import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.encodings.solutionType.IntSolutionType;
import jmetal.util.JMException;
import java.io.*;

public class MEProblem extends Problem {
	public class Scenario {
		public int[] Cores;
		public int[] SSJ;
		public float[] IdleEnergy;
		public float[] MaxEnergy;
	}

	public class Workload {
		public float[][] ETC;
		public float[][] Energy;
	}

	private int taskCount;
	private int machineCount;
	private Scenario scenario;
	private Workload workload;

	public MEProblem(int taskCount, int machineCount, String scenarioPath,
			String workloadPath) throws ClassNotFoundException, IOException {
		this.taskCount = taskCount;
		this.machineCount = machineCount;

		this.scenario = new Scenario();
		this.scenario.Cores = new int[this.machineCount];
		this.scenario.SSJ = new int[this.machineCount];
		this.scenario.IdleEnergy = new float[this.machineCount];
		this.scenario.MaxEnergy = new float[this.machineCount];

		FileInputStream fstream = new FileInputStream(scenarioPath);
		DataInputStream in = new DataInputStream(fstream);
		BufferedReader br = new BufferedReader(new InputStreamReader(in));
		String strLine;

		int currentMachine = 0;
		int currentIndex;
		int currentData;

		int cores = 0;
		int ssj = 0;
		float idle = 0;
		float max = 0;
		String[] scenarioData;

		while ((strLine = br.readLine()) != null) {
			scenarioData = strLine.split(" ");

			currentData = 0;
			currentIndex = 0;
			while ((currentData < 4) && (currentIndex < scenarioData.length)) {
				if (!scenarioData[currentIndex].trim().equals("")) {
					if (currentData == 0) {
						cores = Integer.parseInt(scenarioData[currentIndex]
								.trim());
					} else if (currentData == 1) {
						ssj = Integer.parseInt(scenarioData[currentIndex]
								.trim());
					} else if (currentData == 2) {
						idle = Float.parseFloat(scenarioData[currentIndex]
								.trim());
					} else if (currentData == 3) {
						max = Float.parseFloat(scenarioData[currentIndex]
								.trim());
					}

					currentData++;
				}

				currentIndex++;
			}

			this.scenario.Cores[currentMachine] = cores;
			this.scenario.SSJ[currentMachine] = ssj;
			this.scenario.IdleEnergy[currentMachine] = idle;
			this.scenario.MaxEnergy[currentMachine] = max;
			currentMachine++;
		}
		in.close();

		this.workload = new Workload();
		this.workload.ETC = new float[this.machineCount][this.taskCount];
		this.workload.Energy = new float[this.machineCount][this.taskCount];

		fstream = new FileInputStream(workloadPath);
		in = new DataInputStream(fstream);
		br = new BufferedReader(new InputStreamReader(in));

		currentMachine = 0;
		int currentTask = 0;

		while ((strLine = br.readLine()) != null) {
			this.workload.ETC[currentMachine][currentTask] = Float
					.parseFloat(strLine) / this.scenario.SSJ[currentMachine];
			this.workload.Energy[currentMachine][currentTask] = this.workload.ETC[currentMachine][currentTask]
					* this.scenario.MaxEnergy[currentMachine];

			currentMachine++;
			
			if (currentMachine == this.machineCount) {
				currentTask++;
				currentMachine = 0;
			}
		}
		in.close();

		numberOfVariables_ = taskCount;
		numberOfObjectives_ = 2; /* makespan and energy */
		numberOfConstraints_ = 0;
		problemName_ = "MEProblem";

		upperLimit_ = new double[numberOfVariables_];
		lowerLimit_ = new double[numberOfVariables_];
		for (int var = 0; var < numberOfVariables_; var++) {
			lowerLimit_[var] = 0;
			upperLimit_[var] = machineCount-1;
		} // for

		solutionType_ = new IntSolutionType(this);
	}

	@Override
	public void evaluate(Solution solution) throws JMException {
		float totalEnergyConsumption = 0;

		float makespan = 0;
		float[] computeTime = new float[machineCount];

		int assignedMachine;
		for (int t = 0; t < taskCount; t++) {
			assignedMachine = (int) solution.getDecisionVariables()[t]
					.getValue();

			computeTime[assignedMachine] += workload.ETC[assignedMachine][t];
			if (computeTime[assignedMachine] > makespan)
				makespan = computeTime[assignedMachine];

			totalEnergyConsumption += workload.Energy[assignedMachine][t];
		}

		for (int m = 0; m < machineCount; m++) {
			totalEnergyConsumption += (scenario.IdleEnergy[m] * (makespan - computeTime[m]));
		}

		solution.setObjective(0, makespan);
		solution.setObjective(1, totalEnergyConsumption);
	}

	public void evaluateConstraints(Solution solution) throws JMException {
		/* no constraints defined */
	}
}
