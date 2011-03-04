package AII;

import java.io.*;
import java.util.Iterator;
import java.util.Scanner;
import java.util.TreeMap;

public class CompararResultadosTodos {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			TreeMap<String, Integer> tokens = new TreeMap<String, Integer>();
			TreeMap<String, Integer> tokensErrors = new TreeMap<String, Integer>();
			TreeMap<String, TreeMap<String, Integer>> tokensMatrizConfusion;
			tokensMatrizConfusion = new TreeMap<String, TreeMap<String, Integer>>();
			
			for (int i = 0; i < 10; i++) {
				String corpus = "corpus/test_full_"+i+".txt";
				String salida = "corpus/result_"+i+".txt";
				
				System.out.println(">> Procesando i = " + i);
				
				FileInputStream corpusStream = null;
				Scanner corpusScanner = null;
				corpusStream = new FileInputStream(corpus);
				corpusScanner = new Scanner(corpusStream, "UTF-8");

				FileInputStream salidaStream = null;
				Scanner salidaScanner = null;
				salidaStream = new FileInputStream(salida);
				salidaScanner = new Scanner(salidaStream, "UTF-8");

				String salidaToken;
				String corpusToken;

				int corpusTokenIndex = 0;
				int salidaTokenIndex = 0;

				while (corpusScanner.hasNext()) {
					corpusToken = corpusScanner.next();

					if (corpusTokenIndex % 2 != 0) {
						// System.out.println("[corpus] " + corpusToken);

						if (salidaScanner.hasNext()) {
							salidaToken = salidaScanner.next();
							// System.out.println("[salida] " + salidaToken);

							String currentToken;
							currentToken = corpusToken.trim().toUpperCase();
							if (!tokens.containsKey(currentToken)) {
								tokens.put(currentToken, 0);
								tokensErrors.put(currentToken, 0);
								tokensMatrizConfusion.put(currentToken,
										new TreeMap<String, Integer>());
							}

							tokens.put(currentToken,
									tokens.get(currentToken) + 1);

							if (currentToken.equals(salidaToken.trim()
									.toUpperCase())) {
								// OK!!!
								TreeMap<String, Integer> tokenConfusion = tokensMatrizConfusion
										.get(currentToken);
								if (!tokenConfusion.containsKey(currentToken)) {
									tokenConfusion.put(currentToken, 0);
								}
								tokenConfusion.put(currentToken,
										tokenConfusion.get(currentToken) + 1);
							} else {
								// ERROR!!!
								tokensErrors.put(currentToken,
										tokensErrors.get(currentToken) + 1);
								TreeMap<String, Integer> tokenConfusion = tokensMatrizConfusion
										.get(currentToken);
								if (!tokenConfusion.containsKey(salidaToken)) {
									tokenConfusion.put(salidaToken, 0);
								}
								tokenConfusion.put(salidaToken,
										tokenConfusion.get(salidaToken) + 1);
							}

							salidaTokenIndex++;
						} else {
							System.out.println("Error!!! faltan lineas en "
									+ salida + " ¿?");
							System.exit(-1);
						}
					}

					corpusTokenIndex++;
				}

				System.out.println("Total corpus tokens: " + corpusTokenIndex
						/ 2);
				System.out.println("Total salida tokens: " + salidaTokenIndex);
			}

			Iterator<String> keys = tokens.keySet().iterator();
			while (keys.hasNext()) {
				String token;
				token = keys.next();

				System.out
						.println("===================================================");
				System.out.println(">> Token '" + token + "'");
				System.out.println("Total: " + tokens.get(token));
				System.out.println("Errores: " + tokensErrors.get(token) + " ("
						+ (double)tokensErrors.get(token) / (double)tokens.get(token) * 100.0
						+ "%)");
				System.out.println();
				System.out.println("Matriz de confusión:");

				TreeMap<String, Integer> tokenMatrizConfusion = tokensMatrizConfusion
						.get(token);
				Iterator<String> keysConfusion = tokenMatrizConfusion.keySet()
						.iterator();
				while (keysConfusion.hasNext()) {
					String tokenConfusion;
					tokenConfusion = keysConfusion.next();

					System.out.println("[" + tokenConfusion + ": "
							+ tokenMatrizConfusion.get(tokenConfusion) + "] ");
				}

				System.out
						.println("===================================================");
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}
	}
}
