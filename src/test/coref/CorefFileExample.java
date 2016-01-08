package test.coref;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Properties;

import edu.stanford.nlp.hcoref.CorefCoreAnnotations;
import edu.stanford.nlp.hcoref.data.CorefChain;
import edu.stanford.nlp.hcoref.data.Mention;
import edu.stanford.nlp.ling.CoreAnnotations;
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
 * 对文件进行消解操作
 * 输入文件格式 ： 句子\t 问题mention\t 标准答案
 * @author zkli
 *
 */
public class CorefFileExample {
	public static boolean MODE_DEBUG = false;
	
	public static HashSet<String> lexicons_pronouns = new HashSet<>();
	public static HashSet<String> lexicons_verb_whitelist = new HashSet<>();
	public static HashSet<String> lexicons_punctuations = new HashSet<>();
	
	public static void main(String[] args) throws Exception {
		long startTime=System.currentTimeMillis();
		
		args = new String[] {"-props", "edu/stanford/nlp/hcoref/properties/zh-coref-default.properties" };
		Properties props = StringUtils.argsToProperties(args);
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		initial_lexicon();
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader("test_zkli/gold_ans_pronoun_resolution.txt"));
//			BufferedReader reader = new BufferedReader(new FileReader("test_zkli/gold_ans_np_resolution.txt"));
			String templine = "";
			int linecount = 0;
			while ((templine = reader.readLine()) != null) {
				linecount++;
				System.out.println("\n"+ linecount + "\t" + templine);
				
				String sarray[] = templine.split("\t");
				String text = sarray[0];
				String question = sarray[1];
				
				Annotation document = new Annotation(text);
				pipeline.annotate(document);				
				ArrayList<String> list_word = new ArrayList<>();
				ArrayList<String> list_pos = new ArrayList<>();
				ArrayList<String> mentions = new ArrayList<>();
				
				for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
					ruleBasedNPMentionCombine(sentence, list_word, list_pos);
					
					if(MODE_DEBUG)
					{
						Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
						tree.pennPrint(System.out);
						System.out.println(sentence.get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class).
								toString(SemanticGraph.OutputFormat.LIST));
					}

					for (Mention m : sentence.get(CorefCoreAnnotations.CorefMentionsAnnotation.class)) {
						mentions.add(m.toString().replaceAll(" ", ""));
					}
				}
				
				if(MODE_DEBUG)
				{
					System.out.println("---\ncombine word/pos list");
					for (int i = 0; i < list_word.size(); i++) {
						System.out.println(list_word.get(i) + "\t" + list_pos.get(i));
					}
				}
				
				if(MODE_DEBUG)	System.out.println("---\ncoref chains via CoreNLP");
				ArrayList<String> res_corenlp = new ArrayList<>();
				for (CorefChain cc : document.get(CorefCoreAnnotations.CorefChainAnnotation.class).values()) {
					String filtered_cc = ruleCorefFilter(cc.toString());
					if(filtered_cc.equals(""))	continue;
					if(MODE_DEBUG)	System.out.println("\t" + filtered_cc);
					res_corenlp.add(filtered_cc);
					
				}
				System.out.println(">>>result via CoreNLP");
				String res_stanford = stanfordBasedResolution(res_corenlp, question);
				
				if(MODE_DEBUG)	System.out.println("---\ncoref chains via rules");
				String res_rules = ruleBasedResolution(text, question, mentions, list_word, list_pos);
				
				output_final_result(res_stanford, res_rules);
			}
			reader.close();
		} catch (Exception e) {
				e.printStackTrace();
		}
		
		long endTime=System.currentTimeMillis(); 
		long time = (endTime-startTime)/1000;
		System.out.println("Running time "+time/60+"min "+time%60+"s");
	}
	

	/**
	 * 找出res list中和question相关的mention
	 * @param res_corenlp
	 * @param question
	 * @return
	 */
	private static String stanfordBasedResolution(ArrayList<String> res_corenlp, String question) {
		// TODO Auto-generated method stub
		String res_stanford = "";
		for(String pair : res_corenlp)
		{
			String pronoun_extracted = pair.split("\t")[1];
			if(pronoun_extracted.contains(question) || question.contains(pronoun_extracted))
			{
				System.out.println("\t" + pair);
				res_stanford = pair;	
			}
		}
		return res_stanford;
	}


	/**
	 * 如果stanford消解有结果，则使用stanford的；
	 * 如果没有，则用我们的rules(绝大部分进入这部分)
	 * @param res_stanford
	 * @param res_rules
	 */
	private static void output_final_result(String res_stanford, String res_rules) {
		// TODO Auto-generated method stub
		System.out.println(">>>>>>FINAL ANSWER>>>>>>");
		if(!res_stanford.equals(""))	System.out.println(res_stanford);
		else System.out.println(res_rules);
	}


	/**
	 * 分为 代词指代 和 名词短语 指代
	 * @param templine
	 * @param question 
	 * @param mentions
	 * @param list_word
	 * @param list_pos
	 */
	private static String ruleBasedResolution(String templine, String question, ArrayList<String> mentions, ArrayList<String> list_word,
			ArrayList<String> list_pos) {
		//mention reduce result
		ArrayList<String> mentions_reduce = reduce_mentions(mentions);
		//pronoun resolution
		String res_pronoun = ruleBasedPronounMentionsExtract(templine, question, mentions, mentions_reduce);
		//np resolution
		String res_np = ruleBasedNPMentionsExtract(templine, question, mentions_reduce, list_word, list_pos);
		
		if(!res_pronoun.equals(""))	return res_pronoun;//优先代词指代
		else return res_np;//如果代词指代为空，则为名词短语指代结果
	}

	/**
	 * 因为NP的消解一般 不会有chain产生，所以需要加一些规则
	 * @param templine
	 * @param question 
	 * @param mentions_reduce
	 * @param list_word
	 * @param list_pos
	 */
	private static String ruleBasedNPMentionsExtract(String templine, String question, ArrayList<String> mentions_reduce,
			ArrayList<String> list_word, ArrayList<String> list_pos) 
	{
		//给压缩后的mentions标注上pos信息，这样可以后面用
		ArrayList<String> mentions_reduce_pos = new ArrayList<>();
		for(String mention : mentions_reduce)
		{
			String pos_temp = "";
			int min_count = 1000000;//碰到最长相似度的词就停止
			for(String word : list_word)
				if(mention.contains(word) && Math.abs(word.length()-mention.length())<=min_count)
				{
					min_count = Math.abs(word.length()-mention.length());
					pos_temp = list_pos.get(list_word.indexOf(word));
				}
			if(pos_temp.equals(""))
				mentions_reduce_pos.add("Unk");
			else
				mentions_reduce_pos.add(pos_temp);
		}
		
		if(MODE_DEBUG)	
		{
			System.out.println("---\nmentions pos alignment");
			for (int i = 0; i < mentions_reduce.size(); i++) 
				System.out.println(mentions_reduce.get(i) + "\t" + mentions_reduce_pos.get(i));
		}	
		System.out.println(">>>rules of noun phrase");
		String final_result_mention = "";
		for (int i = 0; i < mentions_reduce.size(); i++) {
			String mention = mentions_reduce.get(i);
			if(mention.contains(question) || question.contains(mention))//mention前后找NN DNP
			{
				int index_before = i-1;
				String mention_before = "";//mention往前找，直到找到NN 和 NP
				while((index_before>-1) && !(mentions_reduce_pos.get(index_before).contains("NN")||mentions_reduce_pos.get(index_before).contains("NP")
						||mentions_reduce_pos.get(index_before).contains("NR")))
					index_before--;
					
				if((index_before>-1) && (mentions_reduce_pos.get(index_before).contains("NN")||mentions_reduce_pos.get(index_before).contains("NP")
						||mentions_reduce_pos.get(index_before).contains("NR")))
				{
					//XX v NN 表明不可能为指代关系，只可能为主谓关系, eg 树好像挂满了小灯笼
					String mention_before_candi_NN = mentions_reduce.get(index_before);
					if(!(have_vv_between_two_mentions(list_word, list_pos, question, mention_before_candi_NN)))
						mention_before = mentions_reduce.get(index_before);
					else
					{
						index_before = index_before -1;
						while((index_before>-1) && !(mentions_reduce_pos.get(index_before).contains("NN")||mentions_reduce_pos.get(index_before).contains("NP")
								||mentions_reduce_pos.get(index_before).contains("NR")))
							index_before--;
						if((index_before>-1) && (mentions_reduce_pos.get(index_before).contains("NN")||mentions_reduce_pos.get(index_before).contains("NP")
								||mentions_reduce_pos.get(index_before).contains("NR")))
							mention_before = mentions_reduce.get(index_before);
					}	
				}
				//前面如果找不见 找后面
				if(mention_before.equals(""))//not found
				{
					int index_after = i+1;
					String mention_after = "";
					while((index_after<mentions_reduce.size()) && !(mentions_reduce_pos.get(index_after).contains("NN")||mentions_reduce_pos.get(index_after).contains("NP")
							||mentions_reduce_pos.get(index_after).contains("NR")))
						index_after++;
					if((index_after<mentions_reduce.size()) && (mentions_reduce_pos.get(index_after).contains("NN")||mentions_reduce_pos.get(index_after).contains("NP")
							||mentions_reduce_pos.get(index_after).contains("NR")))
						mention_after = mentions_reduce.get(index_after);
					final_result_mention = mention_after;//最后如果找不到就是NULL
				}
				else
					final_result_mention = mention_before;
			}
		}
		if(final_result_mention.equals("") || final_result_mention.contains(question))//如果找不到的话，就最近的一个NN代替 
		{
			final_result_mention = "";
			int index_question = list_word.indexOf(question);
			for (int i = index_question-1; i > -1 && final_result_mention.equals(""); i--) //往前找
			{
				if(list_pos.get(i).contains("NN") || list_pos.get(i).contains("NP") || list_pos.get(i).contains("NR"))
					final_result_mention = list_word.get(i);
				if(final_result_mention.contains(question))//如果含有的话
					final_result_mention = "";
			}
			if(final_result_mention.equals(""))//前面没有往后找
			{
				for (int i = index_question+1; i < list_word.size() && final_result_mention.equals(""); i++) //往前找
				{
					if(final_result_mention.contains(question))	continue;
					if(list_pos.get(i).contains("NN") || list_pos.get(i).contains("NP") || list_pos.get(i).contains("NR"))
						final_result_mention = list_word.get(i);
					if(final_result_mention.contains(question))
						final_result_mention = "";
				}
			}	
		}
		String res_np = "";
		if(!final_result_mention.equals(""))
		{
			System.out.println("\t" + final_result_mention + "\t" + question);
			res_np = final_result_mention + "\t" + question;
		}
		return res_np;
	}

	/**
	 * “树上好像挂满了小灯笼” 树 NN 小灯笼 NN  但是中间有个 挂 VV
	 * 所以他们是SBV关系
	 * @param list_word
	 * @param list_pos
	 * @param question
	 * @param mention_before_candi_NN
	 * @return
	 */
	private static boolean have_vv_between_two_mentions(ArrayList<String> list_word, ArrayList<String> list_pos,
			String question, String mention_before_candi_NN) {
		// TODO Auto-generated method stub
		int mention_before_candi_NN_index = findBeforeNearestIndex(list_word, question, mention_before_candi_NN);
		int question_index = 10000;
		for (int i = list_word.size()-1; i > 0; i--) {
			if(question_index>mention_before_candi_NN_index && list_word.get(i).contains(question))
				question_index = i;
		}//找到candidate后面第一个出现question坐标的地方
		boolean have_vv = false;
		boolean have_punc = false;//老猫就会绕着这只小猫不停地转，好像在问：“小宝贝，你哪里不舒服？
		for (int i = mention_before_candi_NN_index+1 ; i<question_index && i<list_word.size(); i++) {
			String temp_pos = list_pos.get(i);
			String temp_word = list_word.get(i);
			if((temp_pos.contains("VV") || temp_pos.contains("BA")) && 
					!lexicons_verb_whitelist.contains(temp_word))	
				have_vv = true;
			if(lexicons_punctuations.contains(temp_word))	
				have_punc = true;
		}
		if(!have_punc && have_vv)	return true;//两个mention中间有标点，证明已经不是一句话了 不是SBV的概率会大一些
		return false;
	}

	/**
	 * 有这么一种情况
	 * >>> 他高兴地喊：“妈妈，你看我的小白兔！” 妈妈看着儿子的作品
	 * 如果直接list_word.indexOf(妈妈) 则定位到前一个NN 我们目标是后一个NN
	 * 这里也不能用lastIndexOf(NN)
	 * 
	 * 所以尝试找最近的一个
	 * @param list_word
	 * @param mention
	 * @param mention_candi_NN
	 * @return
	 */
	private static int findBeforeNearestIndex(ArrayList<String> list_word, String question, String mention_candi_NN) {
		// TODO Auto-generated method stub
		int question_id = 0;
		for (int i = 0; i < list_word.size(); i++) 
			if(list_word.get(i).contains(question))	question_id = i;
		for(int index = question_id; index>-1 ; index--)
			if(list_word.get(index).contains(mention_candi_NN) || mention_candi_NN.contains(list_word.get(index)))	return index;
		return 0;
	}

	/**
	 * mention filter to reduce some bad mention
	 * @param mentions 
	 * @return
	 */
	private static ArrayList<String> reduce_mentions(ArrayList<String> mentions) {
		// TODO Auto-generated method stub
		//mention filter to reduce some bad mention
		ArrayList<String> mentions_reduce = new ArrayList<>();
		for(String mention : mentions)
		{
			if(mention.contains("，"))	continue;//这些 鹅 ， 红 嘴巴 ， 高额头 ，
			if(mentions_reduce.size()>0 && mentions_reduce.get(mentions_reduce.size()-1).contains(mention)) continue; //它 它
//			if(is_contain_pronoun(mention)) continue;//他们 后腿
			mentions_reduce.add(mention);
		}
//		for(String reduces : mentions_reduce)
//		{
//			System.out.println(reduces);
//		}
		return mentions_reduce;
	}

	/**
	 * rules 
	 * 1. speakers
	 *    A对B说 我 你(们)
	 *    A对B说 我(们)
	 *    A对B说 你(们)
	 *    A说 我(们)
	 * 2. pronoun and no say
	 * @param question 
	 * 
	 * @param mentions
	 * @param mentions_reduce2 
	 * @param list_pos 
	 * @param list_word 
	 */
	private static String ruleBasedPronounMentionsExtract(String text, String question, ArrayList<String> mentions, ArrayList<String> mentions_reduce) {
		//1. speaker me
		ArrayList<String> res = new ArrayList<>();
		if(text.contains("说"))
		{
			String head_before_say = text.substring(0, text.indexOf("说"));
			String final_mention = "";
			if(text.contains("对"))//A对B说
			{
				String head_before_to = text.substring(0, text.indexOf("对"));
				if(text.contains("你"))
				{
					final_mention = get_str_between_words(text, "对", "说");
					if(!lexicons_pronouns.contains(final_mention))//先行词不能是代词了
					{
						if(text.contains("你们"))
							res.add(final_mention + "\t你们");
						else
							res.add(final_mention + "\t你");
					}
						
				}
				if(text.contains("我"))
				{
					for(String mention : mentions)
						if(head_before_to.contains(mention))
							final_mention = mention;//A对B说 对  之前的最后一个mention 代表A 我
					if(!lexicons_pronouns.contains(final_mention))
					{
						if(text.contains("我们"))
							res.add(final_mention + "\t我们");
						else
							res.add(final_mention + "\t我");
					}
				}
			}
			else//A说 我
			{
				if(text.contains("我"))
				{
					for(String mention : mentions)
						if(head_before_say.contains(mention))
							final_mention = mention;//说 之前的最后一个mention
					if(!lexicons_pronouns.contains(final_mention))
					{
						if(text.contains("我们"))
							res.add(final_mention + "\t我们");
						else
							res.add(final_mention + "\t我");
					}
				}
			}
		}
		//2.pronoun && no say
		else
		{
			//begin pronoun rule
			int index = 0;
			for(String mention : mentions_reduce)
			{
				//我是雪人 but not  我是我
				if(index==0 && lexicons_pronouns.contains(mention) && mentions_reduce.size()>1 && !lexicons_pronouns.contains(mentions_reduce.get(1)))	
					res.add(mentions_reduce.get(1) + "\t" + mention);
				
				if(index > 0 && lexicons_pronouns.contains(mention) && !lexicons_pronouns.contains(mentions_reduce.get((index-1))))
				{
					String word_front = mentions_reduce.get((index-1));
					if(get_str_between_words(text, word_front, mention).equals("和"))//月亮和我
					{
						if(index > 1 && lexicons_pronouns.contains(mention) && !lexicons_pronouns.contains(mentions_reduce.get((index-2))))
						{
							res.add(mentions_reduce.get((index-2)) + "\t" + mention);
						}
					}
					else//雪人是我
					{
						res.add(word_front + "\t" + mention);
					}
				}
				if(index > 0 && lexicons_pronouns.contains(mention) && lexicons_pronouns.contains(mentions_reduce.get((index-1))))
				{
					int index_temp = index-1;
					while(index_temp>-1 && lexicons_pronouns.contains(mentions_reduce.get(index_temp)))
						index_temp --;
					res.add(mentions_reduce.get(index_temp) + "\t" + mention);
				}
				
				index++;
			}
		}
		
		//output
		HashSet<String> pronoun_deweight = new HashSet<>();
		HashSet<String> chains = new HashSet<>();
		for(String chain : res)
		{
			if(!pronoun_deweight.contains(chain.split("\t")[1]))
			{
				pronoun_deweight.add(chain.split("\t")[1]);
				if(MODE_DEBUG)	System.out.println("\t" + chain.trim());
				chains.add(chain.trim());
			}
		}
		//find answer
		System.out.println(">>>result via rule pronoun");
		String final_pronoun = "";
		for(String pair : chains)
		{
			String pronoun_extracted = pair.split("\t")[1];
			if(pronoun_extracted.contains(question) || question.contains(pronoun_extracted))
			{
				final_pronoun = pair;
				System.out.println("\t" + pair);
			}
		}
		return final_pronoun;
	}
	
	/**
	 * 对stanford官方结果进行修正。。
	 * 1. 将含有逗号的句子 保留前半部分
	 * 2. 前后一样的mention删掉，比如 他们--他们
	 * @param str
	 * @return
	 */
	private static String ruleCorefFilter(String str) {
		// TODO Auto-generated method stub
		String pair = "";
		while(str.indexOf("\"")!=-1)
		{
			String nextstr = str.substring(str.indexOf("\"")+1);
			pair += nextstr.substring(0, nextstr.indexOf("\""))+"\t";
			str = nextstr.substring(nextstr.indexOf("\"")+1);
		}
		String res = "";
		
		// 1. 将含有逗号的句子 保留前半部分
		for(String word : pair.split("\t"))
		{
			int index = word.indexOf("，");
			if(index!=-1)
				res+=word.substring(0,index).replaceAll(" ", "")+"\t";
			else
				res+=word.replaceAll(" ", "")+"\t";
				
		}
		//2. 前后一样的mention删掉，比如 他们--他们
		boolean is_same = true;
		String parts[] = res.split("\t");
		for(int i = 1;i<parts.length; i++)
		{
			if(!parts[i-1].equals(parts[i]))	is_same = false;
		}
		if(!is_same)//not same
			return res.trim();
		else
			return "";
	}
	
	/**
	 * 根据规则抽取一部分NP短语，标记为对应词性，然后辅助mention的词性判断
	 * @param sentence
	 */
	private static void ruleBasedNPMentionCombine(CoreMap sentence, ArrayList<String> list_word, ArrayList<String> list_pos) {
		ArrayList<String> tmp_list_word = new ArrayList<>();
		ArrayList<String> tmp_list_pos = new ArrayList<>();
		
		for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
			String word = token.get(TextAnnotation.class);
			String pos = token.get(PartOfSpeechAnnotation.class);
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
				for(; before_index > -1 && (list_pos.get(before_index).equals("JJ") || list_pos.get(before_index).equals("AD"))
						; before_index--)
				{
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
					
				int count_before = list_word.size()-before_index;//if == 3 delete 2 words ; ==4 delete 3 words
				for(int count = 0; count<count_before-1; count++)
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
	}
	
	/**
	 * pronoun list initial
	 * verb white list initial
	 * punc list initial
	 * @return hashset<String>
	 */
	private static void initial_lexicon() {
		// TODO Auto-generated method stub
		//pronoun lexicons
		String pronouns[] = {"它","它们","他","他们","她","她们","自己","你们","我们","咱们","大家","这里","那里","这儿","那儿","我","你","其"};
		for(String pronoun : pronouns)
			lexicons_pronouns.add(pronoun);
		//verb white list
		String white_list_array[] = {"当做","当成","看做","看成"};
		for(String verb : white_list_array)
			lexicons_verb_whitelist.add(verb);
		//punctuation list
		String punctuations[] = {"，","。","；","！","？"};
		for(String punc : punctuations)
			lexicons_punctuations.add(punc);
	}

	/**
	 * 两个词之间的文本
	 * @param text
	 * @param word_front
	 * @param word_rear
	 * @return
	 */
	private static String get_str_between_words(String text, String word_front, String word_rear) {
		// TODO Auto-generated method stub
		return text.substring(
				text.indexOf(word_front)+word_front.length(), 
				text.indexOf(word_front)+word_front.length()+text.substring(text.indexOf(word_front)+word_front.length()).indexOf(word_rear)
				);
	}
}
