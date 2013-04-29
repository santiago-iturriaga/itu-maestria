//  NSGAIIStudy.java
//
//  Authors:
//       Antonio J. Nebro <antonio@lcc.uma.es>
//       Juan J. Durillo <durillo@lcc.uma.es>
//
//  Copyright (c) 2011 Antonio J. Nebro, Juan J. Durillo
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
// 
//  You should have received a copy of the GNU Lesser General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

package jmetal.experiments;

import java.util.logging.Logger;
import java.io.IOException;
import java.util.HashMap;
import java.util.Properties;
import java.util.logging.Level;

import jmetal.core.Algorithm;
import jmetal.core.Problem;
import jmetal.experiments.settings.NSGAII_Settings;
import jmetal.experiments.settings.OPI_NSGAII_Settings;
import jmetal.experiments.util.Friedman;
import jmetal.experiments.util.RBoxplot;
import jmetal.experiments.util.RWilcoxon;
import jmetal.util.JMException;

/**
 */
public class OPIStudy extends Experiment {
	/**
	 * Configures the algorithms in each independent run
	 * 
	 * @param problem
	 *            The problem to solve
	 * @param problemIndex
	 * @param algorithm
	 *            Array containing the algorithms to run
	 * @throws ClassNotFoundException
	 */
	public synchronized void algorithmSettings(String problemName,
			int problemIndex, Algorithm[] algorithm)
			throws ClassNotFoundException {
		try {
			int numberOfAlgorithms = algorithmNameList_.length;
			HashMap[] parameters = new HashMap[numberOfAlgorithms];

			for (int i = 0; i < numberOfAlgorithms; i++) {
				parameters[i] = new HashMap();
			} // for

			for (int i = 0; i < numberOfAlgorithms; i++) {
				parameters[i].put("crossoverProbability_", 0.9);
			}

			for (int i = 0; i < numberOfAlgorithms; i++) {
				algorithm[i] = new OPI_NSGAII_Settings(problemName)
						.configure(parameters[i]);
			}
		} catch (IllegalArgumentException ex) {
			Logger.getLogger(OPIStudy.class.getName()).log(Level.SEVERE, null,
					ex);
		} catch (IllegalAccessException ex) {
			Logger.getLogger(OPIStudy.class.getName()).log(Level.SEVERE, null,
					ex);
		} catch (JMException ex) {
			Logger.getLogger(OPIStudy.class.getName()).log(Level.SEVERE, null,
					ex);
		}
	} // algorithmSettings

	public static void main(String[] args) throws JMException, IOException {
		OPIStudy exp = new OPIStudy(); // exp = experiment

		exp.experimentName_ = "OPIStudy";
		exp.algorithmNameList_ = new String[] { "NSGAII_OPI" };
		exp.problemList_ = new String[] { "FE-HCSP_8x2_1", "FE-HCSP_8x2_2",
				"FE-HCSP_8x2_3", "FE-HCSP_16x3_1", "FE-HCSP_16x3_2",
				"FE-HCSP_16x3_3", "FE-HCSP_32x4_1", "FE-HCSP_32x4_2",
				"FE-HCSP_32x4_3", "FE-HCSP_512x16_1", "FE-HCSP_512x16_2",
				"FE-HCSP_512x16_3" };
		exp.paretoFrontFile_ = new String[] {};
		exp.indicatorList_ = new String[] {};

		int numberOfAlgorithms = exp.algorithmNameList_.length;

		exp.experimentBaseDirectory_ = "/home/santiago/google-hosting/itu-maestria/svn/trunk/jmetal-opi/output/"
				+ exp.experimentName_;
		exp.paretoFrontDirectory_ = "/home/santiago/google-hosting/itu-maestria/svn/trunk/jmetal-opi/output";
		exp.algorithmSettings_ = new Settings[numberOfAlgorithms];
		exp.independentRuns_ = 10;

		// Run the experiments
		int numberOfThreads;
		exp.runExperiment(numberOfThreads = 1);

		// Generate latex tables (comment this sentence is not desired)
		exp.generateLatexTables();
	} // main
} // NSGAIIStudy

