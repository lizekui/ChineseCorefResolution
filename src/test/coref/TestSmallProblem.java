package test.coref;

public class TestSmallProblem {
	public static void main(String args[]) {
		String str = "CHAIN0-[\"小路 娃娃 伤心地 哭了 。\" in sentence 1, \"她\" in sentence 3]";
		String pair = "";
		while(str.indexOf("\"")!=-1)
		{
			String nextstr = str.substring(str.indexOf("\"")+1);
			pair += nextstr.substring(0, nextstr.indexOf("\""))+"\t";
			str = nextstr.substring(nextstr.indexOf("\"")+1);
		}
		System.out.println(pair);
		String res = "";
		for(String word : pair.split("\t"))
		{
			int index = word.indexOf("，");
			if(index!=-1)
				res+=word.substring(0,index).replaceAll(" ", "")+"\t";
			else
				res+=word.replaceAll(" ", "")+"\t";
				
		}
		System.out.println(res);
	}
}
