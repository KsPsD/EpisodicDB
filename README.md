# EpisodicDB

에이전트 메모리를 위한 OLAP 기반 분석 엔진.

기존 메모리 시스템(Mem0, Zep, Letta)은 **검색 시스템**으로 설계됐다.  
EpisodicDB는 에이전트가 실제로 필요한 **분석 시스템**을 만든다.

```sql
-- "이번 주 가장 많이 실패한 tool은?"
SELECT tool_name, COUNT(*) as failures
FROM tool_calls
WHERE outcome = 'failure'
  AND created_at >= NOW() - INTERVAL '7 days'
GROUP BY tool_name
ORDER BY failures DESC;

-- "지금과 비슷한데 성공했던 에피소드는?"
SELECT e.*, array_cosine_distance(e.context_embedding, ?) as dist
FROM episodes e
WHERE e.status = 'success'
ORDER BY dist ASC
LIMIT 5;
```

## 핵심 아이디어

현존 시스템: `쿼리 → Vector DB(유사도) → Knowledge Graph(관계) → KV Store(조회) → Python에서 합산`

EpisodicDB: `쿼리 → 단일 OLAP 엔진 (벡터 유사도 = SQL 연산자 중 하나)`

## 구조

```
src/
  schema/     — episodes, tool_calls, decisions 테이블 정의
  mcp/        — MCP 서버 (에이전트 → 자기 기억 쿼리 인터페이스)
tests/
docs/
```

## 스택

- **엔진**: DuckDB + vss extension
- **인터페이스**: MCP (Model Context Protocol)
- **언어**: Python

## 상태

`리서치 / 초기 구현 단계`
