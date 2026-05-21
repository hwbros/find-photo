# Google Cloud 설정 가이드

## 1. Google Cloud 프로젝트 생성

1. https://console.cloud.google.com 접속
2. 상단 프로젝트 선택 드롭다운 → **새 프로젝트**
3. 프로젝트 이름: `find-photo` (아무 이름)
4. **만들기** 클릭

## 2. Google Drive API 활성화

1. 좌측 메뉴 → **API 및 서비스** → **라이브러리**
2. 검색창에 `Google Drive API` 입력
3. **Google Drive API** 선택 → **사용 설정**

## 3. OAuth 동의 화면 구성

1. 좌측 메뉴 → **API 및 서비스** → **OAuth 동의 화면**
2. User Type: **외부** 선택 → **만들기**
3. 앱 이름: `find-photo`, 사용자 지원 이메일: 본인 이메일 입력
4. **저장 후 계속** (나머지 단계도 전부 **저장 후 계속**)
5. **테스트 사용자** 단계에서 본인 Google 계정 이메일 추가

## 4. OAuth 클라이언트 ID 생성

1. 좌측 메뉴 → **API 및 서비스** → **사용자 인증 정보**
2. **사용자 인증 정보 만들기** → **OAuth 클라이언트 ID**
3. 애플리케이션 유형: **데스크톱 앱**
4. 이름: `find-photo-desktop` → **만들기**
5. **JSON 다운로드** 버튼 클릭

## 5. credentials.json 배치

다운로드한 JSON 파일을 프로젝트 루트에 `credentials.json` 이름으로 저장:

```
find-photo/
├── credentials.json   ← 여기
├── app/
└── ...
```

> `credentials.json`은 `.gitignore`에 포함되어 있어 Git에 올라가지 않습니다.

## 6. 앱 실행 및 최초 로그인

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 시작
uvicorn app.main:app --reload
```

브라우저에서 http://localhost:8000 접속 → **로그인** 버튼 클릭 → Google 계정 인증

> 최초 로그인 시 "앱이 확인되지 않았습니다" 경고가 나올 수 있습니다.  
> **고급** → **find-photo(으)로 이동** 을 클릭하면 됩니다.
