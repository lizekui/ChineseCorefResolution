package test.coref;

import edu.stanford.nlp.hcoref.CorefCoreAnnotations;
import edu.stanford.nlp.hcoref.data.CorefChain;
import edu.stanford.nlp.hcoref.data.Mention;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;

import java.net.Socket;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;

/**
 * A simple example of Stanford Chinese coreference resolution
 * 
 * When I use originAPI code, using the properties file in path
 * edu/stanford/nlp/hcoref/properties/zh-dcoref-default.properties the code
 * could not run correctly in Chinese.
 * 
 * What I did is extracting the right properties file from
 * stanford-chinese-corenlp-2015-12-08-models.jar and replace
 * edu/stanford/nlp/hcoref/properties/zh-coref-default.properties to our
 * originAPI code which finally run correctly.
 * 
 * @originAPI http://stanfordnlp.github.io/CoreNLP/coref.html
 * @modify_author zkli
 */
public class CorefFullExample {
	public static HashSet<String> lexicons_pronouns = new HashSet<>();
	
	public static void main(String[] args) throws Exception {
		long startTime = System.currentTimeMillis();

		String templine = "到了秋天，小青枣慢慢地变红了，变成了很大很大的红枣。这时，树上好像挂满了圆圆的小灯笼。";
		String question = "小灯笼";
//		lexicons_pronouns = initial_lexicon();
//		ArrayList<String> mentions = new ArrayList<>();
//		mentions.add("我聪明的小宝宝");
//		mentions.add("我");
//		mentions.add("月亮");
//		mentions.add("你");
//		mentions.add("你");
//		mentions.add("它");
//		mentions.add("你");
//		mentions.add("它");
//		mentions.add("它");
//		mentions.add("上面");
//		ruleBasedMentionsExtract(text, mentions);
		args = new String[] {"-props", "edu/stanford/nlp/hcoref/properties/zh-coref-default.properties" };

		Annotation document = new Annotation(templine);
		Properties props = StringUtils.argsToProperties(args);
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		pipeline.annotate(document);
		
		lexicons_pronouns = initial_lexicon();
		
		ArrayList<String> list_word = new ArrayList<>();
		ArrayList<String> list_pos = new ArrayList<>();
		ArrayList<String> mentions = new ArrayList<>();
		
		for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
			ruleBasedNPMentionCombine(sentence, list_word, list_pos);
			
//			Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
//			tree.pennPrint(System.out);
//			System.out.println(sentence.get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class).toString(SemanticGraph.OutputFormat.LIST));

//			System.out.println("---");
//			System.out.println("mentions");
			for (Mention m : sentence.get(CorefCoreAnnotations.CorefMentionsAnnotation.class)) {
				mentions.add(m.toString().replaceAll(" ", ""));
			}
		}
		
		System.out.println("---\ncombine word/pos list");
		for (int i = 0; i < list_word.size(); i++) {
			System.out.println(list_word.get(i) + "\t" + list_pos.get(i));
		}
		
		System.out.println("---");
		System.out.println("coref chains via CoreNLP");
		ArrayList<String> res_corenlp = new ArrayList<>();
		for (CorefChain cc : document.get(CorefCoreAnnotations.CorefChainAnnotation.class).values()) {
			String filtered_cc = ruleCorefFilter(cc.toString());
			if(filtered_cc.equals(""))	continue;
			System.out.println("\t" + filtered_cc);
			res_corenlp.add(filtered_cc);
			
		}
		System.out.println(">>>result via CoreNLP");
		for(String pair : res_corenlp)
		{
			String pronoun_extracted = pair.split("\t")[1];
			if(pronoun_extracted.contains(question) || question.contains(pronoun_extracted))
				System.out.println("\t" + pair);
		}
		
		
		System.out.println("---");
		System.out.println("coref chains via rules");
		ruleBasedResolution(templine, question, mentions, list_word, list_pos);

		long endTime = System.currentTimeMillis();
		long time = (endTime - startTime) / 1000;
		System.out.println("Running time " + time / 60 + "min " + time % 60 + "s");
	}

	/**
	 * 分为 代词指代 和 名词短语 指代
	 * @param templine
	 * @param question 
	 * @param mentions
	 * @param list_word
	 * @param list_pos
	 */
	private static void ruleBasedResolution(String templine, String question, ArrayList<String> mentions, ArrayList<String> list_word,
			ArrayList<String> list_pos) {
		//mention reduce result
		ArrayList<String> mentions_reduce = reduce_mentions(mentions);
		//pronoun resolution
		ruleBasedPronounMentionsExtract(templine, question, mentions, mentions_reduce);
		//np resolution
		ruleBasedNPMentionsExtract(templine, question, mentions_reduce, list_word, list_pos);
	}

	/**
	 * 因为NP的消解一般 不会有chain产生，所以需要加一些规则
	 * @param templine
	 * @param question 
	 * @param mentions_reduce
	 * @param list_word
	 * @param list_pos
	 */
	private static void ruleBasedNPMentionsExtract(String templine, String question, ArrayList<String> mentions_reduce,
			ArrayList<String> list_word, ArrayList<String> list_pos) 
	{
		//给压缩后的mentions标注上pos信息，这样可以后面用
		ArrayList<String> mentions_reduce_pos = new ArrayList<>();
		for(String mention : mentions_reduce)
		{
			String pos_temp = "";
			int min_count = 1000000;//碰到最长相似度的词就停止
			for(String word : list_word)
				if(mention.contains(word) && Math.abs(word.length()-mention.length())<min_count)
				{
					min_count = Math.abs(word.length()-mention.length());
					pos_temp = list_pos.get(list_word.indexOf(word));
				}
					
			mentions_reduce_pos.add(pos_temp);
		}
		
		System.out.println("---\nmentions pos alignment");
		for (int i = 0; i < mentions_reduce.size(); i++) {
			System.out.println(mentions_reduce.get(i) + "\t" + mentions_reduce_pos.get(i));
		}
		System.out.println(">>>rules of noun phrase");
		String res_pair = "";
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
					res_pair = mention_after;//最后如果找不到就是NULL
				}
				else
					res_pair = mention_before;
			}
		}
		System.out.println("\t" + res_pair + "\t" + question);
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
		for (int i = mention_before_candi_NN_index+1 ; i<question_index && i<list_word.size(); i++) {
			if(list_pos.get(i).contains("VV"))	return true;
		}
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
		if(list_word.contains(question))
		{
			for(int index_begin = list_word.indexOf(question); index_begin>-1 ; index_begin--)
				if(list_word.get(index_begin).equals(mention_candi_NN))	return index_begin;
		}
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
	private static void ruleBasedPronounMentionsExtract(String text, String question, ArrayList<String> mentions, ArrayList<String> mentions_reduce) {
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
//					System.out.println(text);
//					System.out.println(text.substring(text.indexOf("对")+1));
//					System.out.println(text.indexOf("对")+1);
//					System.out.println(text.substring(text.indexOf("对")+1).indexOf("说"));
//					System.out.println(text.indexOf("说"));
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
				System.out.println("\t" + chain.trim());
				chains.add(chain.trim());
			}
		}
		//find answer
		System.out.println(">>>result via rule pronoun");
		for(String pair : chains)
		{
			String pronoun_extracted = pair.split("\t")[1];
			if(pronoun_extracted.contains(question) || question.contains(pronoun_extracted))
				System.out.println("\t" + pair);
		}
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
					
				int count_before = list_word.size()-before_index;//if == 3 delete 2 words ; ==4 delete 3 words
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
	}
	
	/**
	 * pronoun list initial
	 * @return hashset<String>
	 */
	private static HashSet<String> initial_lexicon() {
		// TODO Auto-generated method stub
		String lexicon_array[] = {"它","它们","他","他们","她","她们","自己","你们","我们","咱们","大家","这里","那里","这儿","那儿","我","你","其"};
		HashSet<String> lexicons_pronouns = new HashSet<>();
		for(String pronoun : lexicon_array)
			lexicons_pronouns.add(pronoun);
		return lexicons_pronouns;
	}

	
	/**
	 * rules 
	 * 1. speakers
	 *    A对B说 我 你(们)
	 *    A对B说 我(们)
	 *    A对B说 你(们)
	 *    A说 我(们)
	 * 2. pronoun and no say
	 * 
	 * @param mentions
	 */
	private static void ruleBasedMentionsExtract(String text, ArrayList<String> mentions) {
		System.out.println("Entering ruleBasedMentionsExtract...");
		
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
//					System.out.println(text);
//					System.out.println(text.substring(text.indexOf("对")+1));
//					System.out.println(text.indexOf("对")+1);
//					System.out.println(text.substring(text.indexOf("对")+1).indexOf("说"));
//					System.out.println(text.indexOf("说"));
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
			//mention filter to reduce some bad mention
			ArrayList<String> mentions_reduce = new ArrayList<>();
			for(String mention : mentions)
			{
				if(mention.contains("，"))	continue;//这些 鹅 ， 红 嘴巴 ， 高额头 ，
				if(mentions_reduce.size()>0 && mentions_reduce.get(mentions_reduce.size()-1).contains(mention)) continue; //它 它
//				if(is_contain_pronoun(mention)) continue;//他们 后腿
				mentions_reduce.add(mention);
			}
//			for(String reduces : mentions_reduce)
//			{
//				System.out.println(reduces);
//			}
			
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
		HashSet<String> pronoun_deweight = new HashSet<>();
		for(String chain : res)
		{
			if(!pronoun_deweight.contains(chain.split("\t")[1]))
			{
				pronoun_deweight.add(chain.split("\t")[1]);
				System.out.println(chain);
			}
		}
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

	/*
	 * 判断一个mention是不是真包含一个代词 比如 “他们后腿” 这样的mention其实是失败的
	 */
	private static boolean is_contain_pronoun(String mention) {
		// TODO Auto-generated method stub
		boolean contains_pronoun = false;
		for(String pronoun : lexicons_pronouns)
			if(!mention.equals(pronoun) && mention.contains(pronoun))// true contain
				contains_pronoun = true;
		return contains_pronoun;
	}
}
