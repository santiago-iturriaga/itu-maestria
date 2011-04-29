package AII;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;

public class ClasificadorHeuristica {

	public static boolean esAdverbio(String palabra) {
		String palabraLower = palabra.toLowerCase();

		return (palabraLower.equals("cuando") || palabraLower.equals("cuanto")
				|| palabraLower.equals("donde") || palabraLower.equals("como")
				|| palabraLower.equals("adonde") || palabraLower.equals("que"));
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		for (int i = 0; i < 10; i++) {
			String corpus = "corpus/test_" + i + ".txt";
			String salida = "model_bl/result_" + i + ".txt";

			// if (args.length == 2) {
			// corpus = args[0];
			// salida = args[1];
			// } else {
			// System.exit(-1);
			// }

			System.out.println("Corpus: " + corpus);
			System.out.println("Salida: " + salida);

			try {
				FileReader corpusReader = new FileReader(corpus);
				BufferedReader corpusIn = new BufferedReader(corpusReader);

				// Leo todas las líneas a memoria.
				ArrayList<ArrayList<String>> corpusLines = new ArrayList<ArrayList<String>>();
				{
					ArrayList<String> currentLine = new ArrayList<String>();
					String currentToken;

					while ((currentToken = corpusIn.readLine()) != null) {
						if (currentToken.trim().length() == 0) {
							// Fin de oración.
							corpusLines.add(currentLine);
							currentLine = new ArrayList<String>();
						} else {
							// Continúo con la oración actual.
							currentLine.add(currentToken);
						}
					}
				}

				FileWriter salidaWriter = new FileWriter(salida);
				BufferedWriter salidaOut = new BufferedWriter(salidaWriter);

				for (int currentLinePos = 0; currentLinePos < corpusLines
						.size(); currentLinePos++) {
					for (int currentTokenPos = 0; currentTokenPos < corpusLines
							.get(currentLinePos).size(); currentTokenPos++) {
						String currentToken = corpusLines.get(currentLinePos)
								.get(currentTokenPos);

						if (esAdverbio(currentToken)) {
							if (corpusLines.get(currentLinePos)
									.contains("?") || corpusLines.get(currentLinePos)
									.contains("¿")) {
								salidaOut.write("CON_TILDE\n");
							} else {
								salidaOut.write("SIN_TILDE\n");
							}
						} else {
							salidaOut.write("O\n");
						}
					}

					if (currentLinePos < corpusLines.size())
						salidaOut.write("\n");
				}

				salidaOut.flush();
				salidaOut.close();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.exit(-1);
			}
		}
	}
}
