package AII;

import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Scanner;
import java.util.TreeMap;

public class CompararResultadoMejorado {
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String corpus = "";
		String salida = "";

		if (args.length == 2) {
			corpus = args[0];
			salida = args[1];
		} else {
			System.exit(-1);
		}

		System.out.println("Corpus: " + corpus);
		System.out.println("Salida: " + salida);

		try {
			FileReader corpusReader = new FileReader(corpus);
			BufferedReader corpusIn = new BufferedReader(corpusReader);

			ArrayList<String> currentLine = new ArrayList<String>();
			String currentToken;
			
			ArrayList<ArrayList<String>> corpusLines = new ArrayList<ArrayList<String>>();
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

			FileReader salidaReader = new FileReader(salida);
			BufferedReader salidaIn = new BufferedReader(salidaReader);

			currentLine = new ArrayList<String>();
			
			ArrayList<ArrayList<String>> salidaLines = new ArrayList<ArrayList<String>>();
			while ((currentToken = salidaIn.readLine()) != null) {
				if (currentToken.trim().length() == 0) {
					// Fin de oración.
					salidaLines.add(currentLine);
					currentLine = new ArrayList<String>();
				} else {
					// Continúo con la oración actual.
					currentLine.add(currentToken);
				}
			}
	
			for (int currentLinePos = 0; currentLinePos < corpusLines.size(); currentLinePos++) {
				for (int currentTokenPos = 0; currentTokenPos < corpusLines.get(currentLinePos).size(); currentTokenPos++) {
					String salidaCurrentToken = salidaLines.get(currentLinePos).get(currentTokenPos);
					
					String[] corpusCurrentTokenAux = corpusLines.get(currentLinePos).get(currentTokenPos).split(" ");
					String corpusCurrentToken = corpusCurrentTokenAux[1];
					
					if (!salidaCurrentToken.trim().equals(corpusCurrentToken.trim())) {
						// ERROR!!!
						MostrarError(corpusLines.get(currentLinePos), currentTokenPos, salidaCurrentToken);
					}
				}
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}
	}
	
	public static void MostrarError(ArrayList<String> linea, int errorPos, String mistakenWith) {
		for (int pos = 0; pos < linea.size(); pos++) {
			System.out.print(linea.get(pos));
			
			if (pos == errorPos) {
				System.out.print(" --> (" + mistakenWith + ")");
			}
			
			System.out.println();
		}
		System.out.println();
	}
}
