package AII;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.regex.Pattern;

public class DatosTestSet {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String testset = args[0]; //"/home/santiago/eclipse/java-workspace/AAI/CRF_Test.log";

		try {
			FileReader reader = new FileReader(testset);
			BufferedReader in = new BufferedReader(reader);

			Hashtable<String, Integer> previous_1 = new Hashtable<String, Integer>();
			Hashtable<String, Integer> previous_2 = new Hashtable<String, Integer>();
			
			ArrayList<ArrayList<String>> sinTildeTestset = new ArrayList<ArrayList<String>>();
			ArrayList<ArrayList<String>> conTildeTestset = new ArrayList<ArrayList<String>>();
			ArrayList<String> currentLine = new ArrayList<String>();
			String corpusLine;

			int cantidadDeConTilde = 0;
			int cantidadDeSinTilde = 0;
			int cantidadDeConTildeConQE = 0;
			int cantidadDeSinTildeConQE = 0;

			int filePosition = 0;
			boolean tieneSIGN_QE = false;
			boolean tieneCON_TILDE = false;
			boolean tieneSIN_TILDE = false;

			while ((corpusLine = in.readLine()) != null) {
				filePosition++;

				if (corpusLine.trim().length() == 0) {
					// Fin de oración.
					if (tieneCON_TILDE) {
						conTildeTestset.add(currentLine);
						cantidadDeConTildeConQE++;
					} else if (tieneSIN_TILDE) {
						sinTildeTestset.add(currentLine);
						cantidadDeSinTildeConQE++;
					}

					tieneCON_TILDE = false;
					tieneSIN_TILDE = false;
					tieneSIGN_QE = false;
					
					currentLine = new ArrayList<String>();
				} else {
					// Continúo con la oración actual.
					currentLine.add(corpusLine);

					if (corpusLine.indexOf("CON_TILDE") >= 0) {
						tieneCON_TILDE = true;
						cantidadDeConTilde++;
						
						Pattern.compile("WORD=.*")
						
						if (currentLine.size() > 1) {
							
						}
					}
					if (corpusLine.indexOf("SIN_TILDE") >= 0) {
						tieneSIN_TILDE = true;
						cantidadDeSinTilde++;
					}
					if (corpusLine.indexOf("SIGN-QE") >= 0) {
						tieneSIGN_QE = true;
					}
				}
			}
			
			System.out.println("cantidadConTilde: " + cantidadDeConTilde);
			System.out.println("cantidadSinTilde: " + cantidadDeSinTilde);
			System.out.println("cantidadConTildeConQE: " + cantidadDeConTildeConQE);
			System.out.println("cantidadSinTildeConQE: " + cantidadDeSinTildeConQE);

			System.out.println("[SIN_TILDE] ====================================================");
			for (int i=0; i < sinTildeTestset.size(); i++) {
				for (int j=0; j < sinTildeTestset.get(i).size(); j++) {
					System.out.println(sinTildeTestset.get(i).get(j));
				}
				
				System.out.println("\n");
			}
			System.out.println("[CON_TILDE] ====================================================");
			for (int i=0; i < conTildeTestset.size(); i++) {
				for (int j=0; j < conTildeTestset.get(i).size(); j++) {
					System.out.println(conTildeTestset.get(i).get(j));
				}
				
				System.out.println("\n");
			}			
			in.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}
}
