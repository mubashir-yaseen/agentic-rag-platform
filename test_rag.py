import os, json, traceback
from rag_system import RAGSystem

def run():
    try:
        rag = RAGSystem()
    except Exception as e:
        print('INIT_FAIL', e)
        traceback.print_exc()
        return
    print('RAG_INIT_OK')
    # User doc test
    try:
        sample_text = "This is a test user document. It explains that the RAG system answers only from provided context. The embedding dimension is 384."
        chunks = rag.chunk_and_embed_document(sample_text, chunk_size=200, chunk_overlap=50, metadata={'source':'test_user'})
        print('USER_CHUNKS', len(chunks))
        doc_id = rag.insert_document_record(scope='user', file_name='test_user_doc.txt', storage_path='user-documents/test_user_doc.txt')
        print('USER_DOC_ID', doc_id)
        cnt = rag.insert_chunks_record(doc_id, scope='user', chunks_data=chunks)
        print('USER_CHUNKS_INSERTED', cnt)
        res = rag.query_user_document(doc_id, 'What does this document explain about RAG?')
        print('USER_QUERY_ANSWER:', res.get('answer'))
    except Exception as e:
        print('USER_TEST_FAIL')
        traceback.print_exc()
    # Company test - try to find a company
    try:
        comp = rag.get_company_by_symbol_or_name('AAPL')
        if not comp:
            resp = rag.supabase.table('companies').select('*').limit(1).execute()
            comp = resp.data[0] if resp.data else None
        if not comp:
            print('NO_COMPANY_AVAILABLE')
            return
        print('COMPANY_FOUND', comp.get('id'), comp.get('symbol'))
        sample_text_c = f"{comp.get('name')} is a company used for testing. It has financial highlights and risks described."
        chunks_c = rag.chunk_and_embed_document(sample_text_c, chunk_size=200, chunk_overlap=50, metadata={'company':comp.get('id')})
        docc = rag.insert_document_record(company_id=comp.get('id'), scope='company', file_name='test_company_doc.txt', storage_path=f"annual-reports/{comp.get('symbol')}/test_company_doc.txt")
        print('COMP_DOC_ID', docc)
        cntc = rag.insert_chunks_record(docc, company_id=comp.get('id'), scope='company', chunks_data=chunks_c)
        print('COMP_CHUNKS_INSERTED', cntc)
        cres = rag.query_company_documents(comp.get('id'), 'What does the company document say about financial highlights?')
        print('COMP_QUERY_ANSWER:', cres.get('answer'))
    except Exception as e:
        print('COMP_TEST_FAIL')
        traceback.print_exc()

if __name__ == '__main__':
    run()
