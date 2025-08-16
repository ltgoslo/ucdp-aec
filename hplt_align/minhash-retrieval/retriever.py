import gaoya
import pandas as pd
import sys
from fire import Fire
import zstandard
from time import time


class Retriever:
    def _preprocess(self, stext):
        # nonletters = r'[^a-zA-Z]'
        nonletters = r'[_0-9\W]'
        return stext.str.replace(nonletters, '', regex=True)

    
    def preprocess_content(self, fcontent='easy.jsonl'):
        """
        Preprocess content for faster adding to the index.
        To avoid repeated ids, we don't use them, instead use line numbers.
        """
        print('Loading', fcontent)
        df = pd.read_json(fcontent, lines=True)    
        #df = df.set_index('id')  # There are duplicates in full.jsonl!
        df.index += 1  # line numbers starting with 1 to avoid +0=-0 problem
        print('Preprocessing', fcontent)
        s1 = self._preprocess(df.article)
        s2 = self._preprocess(df.headline + ' ' + df.article)
        s2.index *= -1  # negated line numbers correspond to concat(headline,article)
        text_ser = pd.concat([s1, s2])
        print('Dumping to disk')
        text_ser.to_csv(fcontent+'.preproc.tsv', sep='\t')
        text_ser.to_pickle(fcontent+'.preproc.pkl')

        print('Survived chars:')
        survived_chars = set.union(*text_ser.apply(set))
        print(repr(''.join(sorted(survived_chars))))
        print('Removed chars:')
        all_chars = set.union(*df.article.apply(set))
        print(repr(''.join(sorted(all_chars - survived_chars))))

        
    def __init__(self, fcontent=None, ng_min=10, ng_max=10):
        if fcontent is None:
           return
#        import pdb
#        pdb.set_trace()
        text_ser = pd.read_pickle(fcontent)
        n = len(text_ser)
        text_ser.drop(text_ser[text_ser.str.len() < ng_min].index, inplace=True)
        print(f'{n} documents loaded, {(n-len(text_ser))/n*100}% were dropped because they are shorter than ng_min={ng_min} chars', file=sys.stderr)

        index = gaoya.minhash.MinHashStringIndex(hash_size=32, 
                                                     jaccard_threshold=0.5, 
                                                     num_bands=42, 
                                                     band_size=3,
                                                     analyzer='char', 
                                                     lowercase=True, 
                                                     ngram_range=(ng_min,ng_max))
        
        index.par_bulk_insert_docs(text_ser.index, text_ser)
        print(f'{len(text_ser)} documents added to index', file=sys.stderr)
        del text_ser
        self.index = index
        self.ng_min = ng_min

        
    def _retrieve(self, qdf, out=sys.stdout):
        """
        qdf: a DataFrame with 2 columns: 'qid' and 'text'
        """
        if len(qdf) == 0: return
        qser = self._preprocess(qdf.text)
        short_mask = qdf[qser.str.len() < self.ng_min].index
        qser.drop(short_mask, inplace=True)
        qdf.drop(short_mask, inplace=True)
        res = self.index.par_bulk_query(qser, return_similarity=True)
        if not any(r for r in res):
            return  # optimization fot the case of no matches
        qdf['ans'] = res
        qdf = qdf.explode(column='ans', ignore_index=True)  # lists of answers to 1 answer per row
        qdf.dropna(subset='ans', inplace=True)  # empty lists of answers
        qdf['tid'], qdf['sim'] = qdf.ans.str[0], qdf.ans.str[1]
        if len(qdf) > 0:
            qdf[['qid','tid','sim','text']].to_json(out, orient='records',lines=True)
        
        
    def retrieve(self, finp='-', batch_size=10000, text_field='text'):
        """
        Retrieve near-duplicated of texts from stdin (by default) or file. 
        For each query text finds similar texts in the index, prints (qid, tid, sim, text) to stdout, where
        qid and tid are query text and retrieved text identifiers, sim is their Jaccard similarity, text is the 
        retrieved text. 
        NB: It is recommended to use batches: 1K queries => 102ms, 10K queries => 955 ms
        NB: texts containing less than ng_min letters are not indexed or queried because they become an empty set of ngrams
        """
        start_line_num = 1
        with sys.stdin if finp=='-' else zstandard.open(finp,'r') as inp:
            while True:
                st = time()
                df = pd.read_json(inp, nrows=batch_size, lines=True, orient='records', dtype=False)  # dtype=False loads text correctly, otherwise tries infers float/int types sometimes
                if len(df)==0: break  # EOF
                if text_field != 'text':
                    df.rename(columns={text_field:'text'}, inplace=True)
                if finp != '-':
                    df['qid'] = f'{finp}:' + (df.index + start_line_num).astype(str)
                    start_line_num += len(df)
                else:
                    df = df[['index','text']].rename(columns={'index':'qid'})
                df.dropna(subset='text', inplace=True)
                dur1 = time() - st
                st = time()
                self._retrieve(df)
                dur2 = time() - st
                print(f"{dur1+dur2}s per batch of {batch_size} docs, {len(df)} queries after filtering, {dur1/dur2}x longer reading than searching", file=sys.stderr)



Fire(Retriever)
