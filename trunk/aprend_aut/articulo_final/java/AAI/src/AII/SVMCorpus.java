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

public class SVMCorpus {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String corpus = "/home/santiago/eclipse/java-workspace/AAI/corpus.txt";
		String path = "/home/santiago/eclipse/java-workspace/AAI/corpus_svm/";

		if (args.length == 2) {
			corpus = args[0];
			
			path = args[1];
			if (path.charAt(path.length()-1) == '/') {
				path = path.substring(0, path.length()-1);
			}
		}

		System.out.println("Corpus: " + corpus);
	
		try {		
			FileReader reader = new FileReader(corpus);
			BufferedReader in = new BufferedReader(reader);
			
			FileWriter trainWriter;
			BufferedWriter trainBuffWriter;
			
			FileWriter testWriter;
			BufferedWriter testBuffWriter;

			FileWriter testWriterFull;
			BufferedWriter testBuffWriterFull;
			
			String corpusLine;
			int totalCorpusLines = 0;
			
            while ((corpusLine = in.readLine()) != null) {
				if (corpusLine.trim().equals("")) {
					totalCorpusLines++;
				}
            }

            in.close();

    		double fraction = 0.1;
            int linesFraction = (int) Math.round(fraction * totalCorpusLines); 
           
			System.out.println(">> Total lines: " + totalCorpusLines);
			System.out.println(">> Test fraction: " + linesFraction);
			
			HashMap<String, Integer> alfabeto;
			
			for (int i = 0; i < 10; i++) {
				alfabeto = new HashMap<String, Integer>();
				
				reader = new FileReader(corpus);
				in = new BufferedReader(reader);

				trainWriter = new FileWriter(path + "/train_" + i + ".txt",false);
				trainBuffWriter = new BufferedWriter(trainWriter);

				testWriter = new FileWriter(path + "/test_" + i + ".txt",false);
				testBuffWriter = new BufferedWriter(testWriter);

				testWriterFull = new FileWriter(path + "/test_full_" + i + ".txt",false);
				testBuffWriterFull = new BufferedWriter(testWriterFull);
				
				int start_offset = i * linesFraction;
				int end_offset;
				if (i < 9) {
					end_offset = start_offset + linesFraction;
				} else {
					end_offset = totalCorpusLines;
				}
				
				System.out.println("=====================================");
				System.out.println(">> Index: " + i);
				System.out.println(">> Start offset: " + start_offset);
				System.out.println(">> End offset: " + end_offset);
				
				ArrayList<String> currentLine = new ArrayList<String>();
				int corpusLineIndex = 0;
				int currentWordIndex = 0;
				
				int exampleLine = 1;
				int testLine = 1;
				int prevWordTag = -1, prevPrevWordTag = -1; 
				
	            while ((corpusLine = in.readLine()) != null) {
	            	currentLine.add(corpusLine);
	            	
					if (corpusLine.trim().equals("")) {
						if ((corpusLineIndex >= start_offset)&&(corpusLineIndex <= end_offset)) {
							System.out.println(">> " + currentLine.size() + " a test");
							
							for (int j = 0; j < currentLine.size(); j++) {
								String tokens;
								tokens = currentLine.get(j).trim();
								
								String[] tokenArray;
								tokenArray = tokens.split(" ");

								String word;
								String tag;
								Integer tagIndex = -1;
								
								if (tokenArray.length == 2) {
									word = tokenArray[0].trim().toLowerCase();
									tag = tokenArray[1].trim().toUpperCase();

									if (tag.equals("O")) {
										tagIndex = 1;
									} else if (tag.equals("SIN_TILDE")) {
										tagIndex = 2;
									} else if (tag.equals("CON_TILDE")) {
										tagIndex = 3;
									}

									if (!alfabeto.containsKey(word)) {
										alfabeto.put(word, currentWordIndex);
										currentWordIndex++;
									}
									
									String nGram = "";
//									if (prevWordTag > 0) {
//										nGram += " 2:" + prevWordTag;
//										
//										if (prevPrevWordTag > 0) {
//											nGram += " 3:" + prevPrevWordTag;
//										}
//									}
									
									testBuffWriter.write(tagIndex + " qid:" + testLine + " 1:" + alfabeto.get(word) + nGram + " # " + word + "\n");
									testBuffWriterFull.write(tagIndex + "\n");
									
									prevPrevWordTag = prevWordTag;
									prevWordTag = alfabeto.get(word);
								} else {
									prevWordTag = -1;
									prevPrevWordTag = -1;
									
									testLine++;										
								}
							}
						} else {
							System.out.println(">> " + currentLine.size() + " a train");
							
							for (int j = 0; j < currentLine.size(); j++) {
								String[] tokenArray;
								tokenArray = currentLine.get(j).split(" ");
								
								String word;
								String tag;
								Integer tagIndex = -1;
								
								if (tokenArray.length >= 2) {
									word = tokenArray[0].trim().toLowerCase();
									tag = tokenArray[1].trim().toUpperCase();

									if (tag.equals("O")) {
										tagIndex = 1;
									} else if (tag.equals("SIN_TILDE")) {
										tagIndex = 2;
									} else if (tag.equals("CON_TILDE")) {
										tagIndex = 3;
									}
									
									if (!alfabeto.containsKey(word)) {
										alfabeto.put(word, currentWordIndex);
										currentWordIndex++;
									}

									String nGram = "";
//									if (prevWordTag > 0) {
//										nGram += " 2:" + prevWordTag;
//										
//										if (prevPrevWordTag > 0) {
//											nGram += " 3:" + prevPrevWordTag;
//										}
//									}
									
									trainBuffWriter.write(tagIndex + " qid:" + exampleLine + " 1:" + alfabeto.get(word) + nGram + " # " + word + "\n");
									
									prevPrevWordTag = prevWordTag;
									prevWordTag = alfabeto.get(word);
								} else {
									prevWordTag = -1;
									prevPrevWordTag = -1;
									
									exampleLine++;
								}
							}
						}
						
						corpusLineIndex++;
						currentLine = new ArrayList<String>();
					}
	            }
	            
	            testBuffWriter.close();
	            testBuffWriterFull.close();
	            trainBuffWriter.close();
	            in.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

}
