package AII;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class DatosTestSet {

	public static String getWord(String token) {
		return token.split(" ")[0].toLowerCase();
	}

	public static Hashtable<String, Integer> GetTopN(
			Hashtable<String, Integer> ocurrencias) {
		int n = 10;
		Hashtable<String, Integer> topN = new Hashtable<String, Integer>();

		String min = null;

		Iterator<String> iter = ocurrencias.keySet().iterator();
		while (iter.hasNext()) {
			String key = iter.next();

			if (topN.size() < n) {
				topN.put(key, ocurrencias.get(key));

				if (topN.size() == n) {
					min = topN.keySet().toArray()[0].toString();

					for (int i = 1; i < n; i++) {
						String aux_key = topN.keySet().toArray()[i].toString();

						if (topN.get(aux_key) < topN.get(min)) {
							min = aux_key;
						}
					}
				}
			} else {
				if (ocurrencias.get(key) > topN.get(min)) {
					topN.remove(min);
					topN.put(key, ocurrencias.get(key));

					min = topN.keySet().toArray()[0].toString();

					for (int i = 1; i < n; i++) {
						String aux_key = topN.keySet().toArray()[i].toString();

						if (topN.get(aux_key) < topN.get(min)) {
							min = aux_key;
						}
					}
				}
			}
		}

		return topN;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String testset_template = "/home/santiago/eclipse/java-workspace/AAI/corpus/test_full_";
		String testset;

		for (int index_test = 0; index_test < 10; index_test++) {
			int cantidadCONCON = 0;
			int cantidadCONSIN = 0;
			int cantidadSINCON = 0;
			int cantidadSINSIN = 0;

			Hashtable<String, Integer> previous_1 = new Hashtable<String, Integer>();
			Hashtable<String, Integer> previous_2 = new Hashtable<String, Integer>();
			Hashtable<String, Integer> next_1 = new Hashtable<String, Integer>();

			ArrayList<ArrayList<String>> sinTildeTestset = new ArrayList<ArrayList<String>>();
			ArrayList<ArrayList<String>> conTildeTestset = new ArrayList<ArrayList<String>>();

			ArrayList<String> currentLine = new ArrayList<String>();
			ArrayList<ArrayList<String>> lines = new ArrayList<ArrayList<String>>();
			ArrayList<String> raw = new ArrayList<String>();
			String currentToken;

			int cantidadDeConTilde = 0;
			int cantidadDeSinTilde = 0;
			int cantidadDeConTildeConQE = 0;
			int cantidadDeSinTildeConQE = 0;

			System.out
					.println("=================================================================================");
			System.out.println("Test set: " + index_test);
			System.out
					.println("=================================================================================");

			for (int index_train = 0; index_train < 10; index_train++) {
				if (index_train != index_test) {
					testset = testset_template + index_train + ".txt";

					try {
						FileReader reader = new FileReader(testset);
						BufferedReader in = new BufferedReader(reader);

						boolean tieneSIGN_QE = false;
						boolean tieneCON_TILDE = false;
						boolean tieneSIN_TILDE = false;

						while ((currentToken = in.readLine()) != null) {
							raw.add(currentToken);
						}

						for (int i = 0; i < raw.size(); i++) {
							currentToken = raw.get(i);

							if (currentToken.trim().length() == 0) {
								// Fin de oración.
								lines.add(currentLine);
								currentLine = new ArrayList<String>();
							} else {
								// Continúo con la oración actual.
								currentLine.add(currentToken);
							}
						}

						String lastToken = "";

						for (int i = 0; i < lines.size(); i++) {
							lastToken = "";

							for (int j = 0; j < lines.get(i).size(); j++) {
								currentToken = lines.get(i).get(j);

//								if (currentToken.indexOf("CON_TILDE") >= 0) {
//									if (lastToken.indexOf("CON_TILDE") >= 0)
//										cantidadCONCON++;
//									if (lastToken.indexOf("SIN_TILDE") >= 0)
//										cantidadCONSIN++;
//
//									tieneCON_TILDE = true;
//									cantidadDeConTilde++;
//								}

								if (currentToken.indexOf("SIN_TILDE") >= 0) {
									String palabra = getWord(currentToken);

									if (j > 1) {
										// -1
										String palabra_contexto = getWord(lines
												.get(i).get(j - 1));

										int conteo = 0;
										if (previous_1
												.containsKey(palabra_contexto)) {
											conteo = previous_1
													.get(palabra_contexto);
										}
										conteo++;
										previous_1
												.put(palabra_contexto, conteo);
									}

									// if (j > 2) {
									// // -2
									// String palabra_contexto = getWord(lines
									// .get(i).get(j - 2));
									//
									// int conteo = 0;
									// if (previous_2
									// .containsKey(palabra_contexto)) {
									// conteo = previous_2
									// .get(palabra_contexto);
									// }
									// conteo++;
									// previous_2
									// .put(palabra_contexto, conteo);
									// }

									if (j + 1 < lines.get(i).size()) {
										// +1
										String palabra_contexto = getWord(lines
												.get(i).get(j + 1));

										int conteo = 0;
										if (next_1
												.containsKey(palabra_contexto)) {
											conteo = next_1
													.get(palabra_contexto);
										}
										conteo++;
										next_1.put(palabra_contexto, conteo);
									}
								}
//								if (currentToken.indexOf("SIN_TILDE") >= 0) {
//									if (lastToken.indexOf("CON_TILDE") >= 0)
//										cantidadSINCON++;
//									if (lastToken.indexOf("SIN_TILDE") >= 0)
//										cantidadSINSIN++;
//
//									tieneSIN_TILDE = true;
//									cantidadDeSinTilde++;
//								}
//								if (currentToken.indexOf("SIGN-QE") >= 0) {
//									tieneSIGN_QE = true;
//								}

								lastToken = currentToken;
							}

							// Fin de oración.
							if (tieneCON_TILDE) {
								conTildeTestset.add(lines.get(i));

								if (tieneSIGN_QE)
									cantidadDeConTildeConQE++;
							} else if (tieneSIN_TILDE) {
								sinTildeTestset.add(lines.get(i));

								if (tieneSIGN_QE)
									cantidadDeSinTildeConQE++;
							}

							tieneCON_TILDE = false;
							tieneSIN_TILDE = false;
							tieneSIGN_QE = false;
						}

						in.close();
					} catch (Exception e) {
						e.printStackTrace();
						System.exit(-1);
					}
				}
			}
			// System.out
			// .println("[SIN_TILDE] ====================================================");
			// for (int i = 0; i < sinTildeTestset.size(); i++) {
			// for (int j = 0; j < sinTildeTestset.get(i).size();
			// j++) {
			// System.out.println(sinTildeTestset.get(i).get(j));
			// }
			//
			// System.out.println("\n");
			// }
			// System.out
			// .println("[CON_TILDE] ====================================================");
			// for (int i = 0; i < conTildeTestset.size(); i++) {
			// for (int j = 0; j < conTildeTestset.get(i).size();
			// j++) {
			// System.out.println(conTildeTestset.get(i).get(j));
			// }
			//
			// System.out.println("\n");
			// }
			System.out
					.println("[ PREV -1 ] ====================================================");
			{
				Hashtable<String, Integer> aux = GetTopN(previous_1);

				Iterator<String> iter = aux.keySet().iterator();
				while (iter.hasNext()) {
					String key = iter.next();
					if (aux.get(key) > 1) {
//						System.out.println(key + ":" + aux.get(key));
						System.out.print(key + "|");
					}
				}
				System.out.println();
			}

			/*
			 * System.out .println(
			 * "[ PREV -2 ] ===================================================="
			 * ); { Hashtable<String, Integer> aux = GetTopN(previous_1);
			 * 
			 * Iterator<String> iter = previous_2.keySet().iterator(); while
			 * (iter.hasNext()) { String key = iter.next(); if
			 * (previous_2.get(key) > 1) { System.out.println(key + ":" +
			 * previous_2.get(key)); } } }
			 */
			System.out
					.println("[ NEXT +1 ] ====================================================");
			{
				Hashtable<String, Integer> aux = GetTopN(next_1);

				Iterator<String> iter = aux.keySet().iterator();
				while (iter.hasNext()) {
					String key = iter.next();
					if (aux.get(key) > 1) {
//						System.out.println(key + ":" + aux.get(key));
						System.out.print(key + "|");
					}
				}
				System.out.println();
			}
			System.out
					.println("================================================================");
			System.out.println("cantidadConTilde: " + cantidadDeConTilde);
			System.out.println("cantidadSinTilde: " + cantidadDeSinTilde);
			System.out.println("cantidadConTildeConQE: "
					+ cantidadDeConTildeConQE);
			System.out.println("cantidadSinTildeConQE: "
					+ cantidadDeSinTildeConQE);

			System.out.println("cantidadConCon: " + cantidadCONCON);
			System.out.println("cantidadConSin: " + cantidadCONSIN);
			System.out.println("cantidadSinCon: " + cantidadSINCON);
			System.out.println("cantidadSinSin: " + cantidadSINSIN);
		}
	}
}
