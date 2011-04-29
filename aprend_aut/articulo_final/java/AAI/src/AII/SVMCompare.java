package AII;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Scanner;
import java.util.TreeMap;

public class SVMCompare {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String resultado = "/home/santiago/eclipse/java-workspace/AAI/corpus_svm/result_0.txt";
		String original = "/home/santiago/eclipse/java-workspace/AAI/corpus_svm/test_full_0.txt";

		if (args.length == 2) {
			original = args[0];
			resultado = args[1];
		} else {
			System.exit(-1);
		}

		System.out.println("Original: " + original);
		System.out.println("Resultado: " + resultado);

		try {
			FileReader original_reader = new FileReader(original);
			BufferedReader original_in = new BufferedReader(original_reader);

			FileReader resultado_reader = new FileReader(resultado);
			BufferedReader resultado_in = new BufferedReader(resultado_reader);

			String originalLine, resultadoLine;
			int currentLine = 0;
			int cantOK_O = 0, cantOK_SIN_TILDE = 0, cantOK_CON_TILDE = 0;
			int cantERROR_O = 0, cantERROR_SIN_TILDE = 0, cantERROR_CON_TILDE = 0;

			while ((originalLine = original_in.readLine()) != null) {
				currentLine++;
				resultadoLine = resultado_in.readLine();

				originalLine = originalLine.trim();
				resultadoLine = resultadoLine.trim();

				boolean error = false;
				if (originalLine.equals("1")) {
					if (!originalLine.equals(resultadoLine)) {
						error = true;
						cantERROR_O++;
					} else {
						error = false;
						cantOK_O++;
					}
				} else if (originalLine.equals("2")) {
					if (!originalLine.equals(resultadoLine)) {
						error = true;
						cantERROR_SIN_TILDE++;
					} else {
						error = false;
						cantOK_SIN_TILDE++;
					}
				} else if (originalLine.equals("3")) {
					if (!originalLine.equals(resultadoLine)) {
						error = true;
						cantERROR_CON_TILDE++;
					} else {
						error = false;
						cantOK_CON_TILDE++;
					}
				}

				if (error) {
					System.out.println("[DIFFERENCIA] Línea: " + currentLine
							+ "\n");
				}
			}

			System.out.println("Cantidad O         => OK:" + cantOK_O
					+ " ERROR:" + cantERROR_O + "\n");
			System.out.println("Cantidad SIN_TILDE => OK:" + cantOK_SIN_TILDE
					+ " ERROR:" + cantERROR_SIN_TILDE + "\n");
			System.out.println("Cantidad CON_TILDE => OK:" + cantOK_CON_TILDE
					+ " ERROR:" + cantERROR_CON_TILDE + "\n");

			original_in.close();
			resultado_in.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}
}
