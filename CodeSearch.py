"""
Example Usage:
import CodeSearch as CS

embedding_path = 'data/stackoverflow_code_search_csv/final_embedding.tsv'
codebase_path = 'data/stackoverflow_code_search_csv/15_19_Clean_Data.csv'
codebase, word_embedding, document_embeddings =CS.code_search_init(embedding_path, codebase_path)

question = "this is a user input quesiton."
document_index, distance=CS.get_most_relevant_document(question, word_embedding, document_embeddings)
result = CS.get_snippet_results(document_index, codebase)

# for faiss
#questions = ["this is a user input quesiton."] 
#document_index, distance=CS.get_most_relevant_document_faiss(questions, word_embedding, document_embeddings)
#result = CS.get_snippet_results_faiss(document_index, codebase)

"""
import json
import csv
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss

def read_codebase(file_path, aws=False):
  """ Read in Stack Overflow data as codebase """
  if aws:

  else:
    codebase=[]
    with open(file_path) as f:
      csv_reader = csv.reader(f, delimiter=',')
      
      next(csv_reader)
      for row in csv_reader:
        temp_dict = {}
        temp_dict["question_id"] = row[0]
        temp_dict["question_tokens"] = nltk.word_tokenize(row[2].lower())
        temp_dict["snippet_tokens"] = nltk.word_tokenize(row[4])
        temp_dict["snippet"] = row[4].replace("\n", "<br>")
        codebase.append(temp_dict) 
  return codebase

def load_embeddings(embeddings_path, aws=False):
  """Loads pre-trained word embeddings from tsv file.

  Args:
    embeddings_path - path to the embeddings file.

  Returns:
    embeddings - dict mapping words to vectors;
    dim - dimension of the vectors.
  """
  if aws:
    embeddings = {}
    reader = csv.reader(embeddings_path, delimiter='\t')
    for line in reader:
      word = line[0]
      embedding = np.array(line[1:]).astype(np.float32)
      embeddings[word] = embedding
    dim = len(line) - 1
  else:
    embeddings = {}
    
    with open(embeddings_path, newline='') as embedding_file:
      reader = csv.reader(embedding_file, delimiter='\t')
      for line in reader:
        word = line[0]
        embedding = np.array(line[1:]).astype(np.float32)
        embeddings[word] = embedding
      dim = len(line) - 1
  
  return embeddings, dim

def get_most_relevant_document(question, word_embedding, document_embeddings, num=10):
  """Return the functions that are most relevant to the natual language question.

  Args:
      question: A string. A Question from StackOverflow. 
      word_embedding: Word embedding generated from codebase.
      doc_embedding: Document embedding generated from codebase
      num: The number of top similar functions to return.

  Returns:
      A list of indices of the top NUM related functions to the QUESTION in the WORD_EMBEDDING.
  
  """
  # convert QUESTION to a vector
  tokenized_ques=question.split()
  vec_ques=np.zeros((1,document_embeddings.shape[1])) #vocab_size
  token_count=0

  for token in tokenized_ques:
    if token in word_embedding:
      vec_ques+=word_embedding[token]
      token_count+=1
  
  if token_count>0:
    mean_vec_ques=vec_ques/token_count


    # compute similarity between this question and each of the source code snippets
    cosine_sim=[]
    for idx, doc in enumerate(document_embeddings):
      #[TODO] fix dimension

      try:
        cosine_sim.append(cosine_similarity(mean_vec_ques, doc.reshape(1, -1))[0][0])
      except ValueError:
        print(question)
        print(vec_ques, token_count)
        print(mean_vec_ques)
        print(doc.reshape(1, -1))
    # get top `num` similar functions
    result_func_id=np.array(cosine_sim).argsort()[-num:][::-1]
    result_similarity=np.sort(np.array(cosine_sim))[-num:][::-1]
  else:
    result_func_id=np.nan
    result_similarity=np.nan
  return result_func_id, result_similarity

def get_most_relevant_document_faiss(questions, word_embedding, doc_embedding, num=10):
  """Return the functions that are most relevant to the natual language question.

  Args:
      questions: An array, each element is a question from StackOverflow. 
      word_embedding: Word embedding generated from codebase.
      doc_embedding: Document embedding generated from codebase
      num: The number of top similar functions to return.

  Returns:
      A list of indices of the top NUM related functions to the QUESTION in the WORD_EMBEDDING.
  
  """
  invalid_ques_id=[]
  vec_ques=np.zeros((len(questions),doc_embedding.shape[1])) #vocab_size
  # convert QUESTION to a vector
  for idx, ques in enumerate(questions):
    tokenized_ques=ques.split()
      
      
  #tokenized_ques=question.split()
  
    token_count=0
    for token in tokenized_ques:
      if token in word_embedding:
          vec_ques[idx]+=word_embedding[token]
          token_count+=1
            
    if token_count>0:
      mean_vec_ques=vec_ques/token_count
    else:
      #none of the tokens in this question exists in our trained wordembedding.
      invalid_ques_id.append(idx)
      
  d = doc_embedding.shape[1]                           # dimension
  nb = doc_embedding.shape[0]                      # database size
  nq = len(questions)                       # nb of queries
  np.random.seed(1234)             # make reproducible
  xb = doc_embedding.astype(np.float32)
  xq = vec_ques.astype(np.float32)
  #print("xq={}".format(xq.shape))
  index = faiss.IndexFlatL2(d)   # build the index
  #print(index.is_trained)
  index.add(xb)                  # add vectors to the index
  #print(index.ntotal)

  k = 10                          # we want to see 4 nearest neighbors

  # sanity check: for the first 5 documents, the most similar one is themselves
  #D, I = index.search(xb[:5], k) 
  #print(I)
  #print(D)
  #print("xb[:5]={}".format((xb[:5]).shape))
  
  D, I = index.search(xq, k)     # actual search
  #print(I)                   # neighbors of the 5 first queries
  #print(D)                  # neighbors of the 5 last queries

  return I, D

def code_search_init(embedding_path, codebase_path):
  """ Compute document embeddings for all functions in codebase.

  Args:
    embedding_path: path of pre-trained word embedding
    codebase_path: path of Github codebase file

  Returns:
    codebase: a list of dictionaries which contain question_id, question_tokens, snippet_tokens, snippet
    word_embedding: Pre-trained word embedding
    document_embeddings: Document embeddings for all functions in codebase
  """
  codebase=read_codebase(codebase_path) # [{"question_id": int, "intent_tokens": [...]}, ...]
  word_embedding, dim = load_embeddings(embedding_path)

  train_size=len(codebase)
  document_embeddings=np.zeros((train_size, dim))

  for idx, example in enumerate(codebase):
    doc_vec_sum=np.zeros(dim)
    for term in example["snippet_tokens"]:
      if term in word_embedding:
        doc_vec_sum+=word_embedding[term]
            
    document_embeddings[idx]=doc_vec_sum/len(example["snippet_tokens"])

  return codebase, word_embedding, document_embeddings

def get_snippet_results_faiss(document_index, codebase):
  result=[]
  for query in document_index:
    tmp=[]
    for snippet_index in query:
      tmp.append(codebase[snippet_index]["snippet"])
    result.append(tmp)
  return result

def get_snippet_results(document_index, codebase):
  result=[]
  for snippet_index in document_index:
    result.append(codebase[snippet_index]["snippet"])
  return result






