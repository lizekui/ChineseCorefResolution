package test.coref;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;

/**
 * parser tree
 * @modify_author zkli
 */
public class ParserTreeExample {
	public static void main(String[] args) throws Exception {
		long startTime = System.currentTimeMillis();

		String text = "到了秋天，小青枣慢慢地变红了，变成了很大很大的红枣。这时，树上好像挂满了圆圆的小灯笼。";
		args = new String[] { "-props", "edu/stanford/nlp/hcoref/properties/zh-parser-default.properties" };

		Annotation document = new Annotation(text);
		Properties props = StringUtils.argsToProperties(args);
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		pipeline.annotate(document);
		
		for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
			
			ruleBasedNPMentionExtractor(sentence);
			
			Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
			tree.pennPrint(System.out);
			
			System.out.println(sentence.get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class)
					.toString(SemanticGraph.OutputFormat.LIST));
		}

		long endTime = System.currentTimeMillis();
		long time = (endTime - startTime) / 1000;
		System.out.println("Running time " + time / 60 + "min " + time % 60 + "s");
	}

	private static void ruleBasedNPMentionExtractor(CoreMap sentence) {
		ArrayList<String> tmp_list_word = new ArrayList<>();
		ArrayList<String> tmp_list_pos = new ArrayList<>();
		ArrayList<String> list_word = new ArrayList<>();
		ArrayList<String> list_pos = new ArrayList<>();
		
		for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
			String word = token.get(TextAnnotation.class);
			String pos = token.get(PartOfSpeechAnnotation.class);
//			String ne = token.get(NamedEntityTagAnnotation.class);
//			System.out.println(word+"\t"+pos+"\t"+ne);
			tmp_list_word.add(word);
			tmp_list_pos.add(pos);
		}
		for (int i = 0; i < tmp_list_word.size(); i++) {
			String word = tmp_list_word.get(i);
			String pos = tmp_list_pos.get(i);
			list_word.add(word);
			list_pos.add(pos);
			if(pos.contains("DEG") || pos.contains("DEC"))
			{
				String word_before_de = "";//before de
				//找到XX的XX的前门的边界，直到非JJ AD eg 很大很大的红枣
				if(list_word.size()-2>-1)
					word_before_de = list_word.get(list_word.size()-2);
				int before_index = list_word.size()-3;
				for(; before_index > -1; before_index--)
				{
					if(!list_pos.get(before_index).equals("JJ") && !list_pos.get(before_index).equals("AD"))
						break;		
					word_before_de = list_word.get(before_index) + word_before_de;
				}
				
				String word_after_de = "";
				//找到XX的XX的后面的边界，直到NN eg 红红的小灯笼
				int after_index = i+1;
				for(boolean is_NN_range = false;(after_index < tmp_list_word.size() && !is_NN_range); after_index++)
				{
					word_after_de += tmp_list_word.get(after_index);
					if(tmp_list_pos.get(after_index).equals("NN"))
						is_NN_range = true;						
				}
					
				int count_before = i-before_index;//if == 3 delete 2 words ; ==4 delete 3 words
				for(int count = 0; count<count_before; count++)
				{
					if(list_word.size()-1 > -1)
					{
						list_word.remove(list_word.size()-1);
						list_pos.remove(list_pos.size()-1);
					}
				}
				list_word.add(word_before_de+word+word_after_de);
				list_pos.add("DNP");
				i+=(after_index-i-1);//往前跳N个词数
			}
		}
		
		for (int i = 0; i < list_word.size(); i++) {
			System.out.println(list_word.get(i) + "\t" + list_pos.get(i));
		}
	}
}
