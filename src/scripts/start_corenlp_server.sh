# Run this from the directory of corenlp jar
# Example Path for a Maven installation
# ~/.m2/repository/edu/stanford/nlp/stanford-corenlp/3.9.2
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
