"""Memoize immutable validated row versions inside local_upsert batches only.

local_upsert replaces row dictionaries, never mutates their contained lists.
Cache hits return fresh link lists, preserving original normalize semantics.
Final import and verify use the original validator, independently rechecking all.
"""
from .register_task18d_result import StagedMirror

class ValidatedBatchMirror(StagedMirror):
    def __init__(self):
        super().__init__()
        self._original_normalize=self.contract.normalize
        self._normalized_rows={}
        self._code_indices={}
        codes={k:self.field_id(k,'Codice stabile') for k in self.contract.tables}
        def normalize(table,values,*,partial=False):
            key=self.contract.key(table)
            code=values.get(codes[key])
            if partial or not code:
                return self._original_normalize(table,values,partial=partial)
            cachekey=(key,code)
            cached=self._normalized_rows.get(cachekey)
            if cached is not None and (values is cached[0] or values is cached[1]):
                return {k:list(v) if isinstance(v,list) else v for k,v in cached[1].items()}
            result=self._original_normalize(table,values,partial=False)
            self._normalized_rows[cachekey]=(values,result)
            return result
        self.contract.normalize=normalize

    def _validated_batch_code_lookup(self,key,records,field,code):
        index=self._code_indices.get(key)
        if index is None:
            index={}
            for row in records:
                value=self.contract.normalize(key,row.get('cellValuesByFieldId',{}))[field]
                index.setdefault(value,[]).append(row)
            self._code_indices[key]=index
        return index.get(code,[])

    def local_upsert(self,table,values,record_id=None):
        result=super().local_upsert(table,values,record_id)
        key=self.contract.key(table)
        index=self._code_indices[key]
        fid=self.field_id(key,'Codice stabile')
        code=self.contract.normalize(key,values,partial=True).get(fid)
        if code is None:
            rows=self.staged['tables'][self.contract.tables[key]['id']]
            row=next(row for row in rows if row['id']==result['record_id'])
            code=row['cellValuesByFieldId'][fid]
        if code not in index:
            rows=self.staged['tables'][self.contract.tables[key]['id']]
            row=next(row for row in reversed(rows) if row['id']==result['record_id'])
            index[code]=[row]
        return result

    def commit_batch(self):
        self.contract.normalize=self._original_normalize
        self._normalized_rows.clear()
        self._code_indices.clear()
        return super().commit_batch()
