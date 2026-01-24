# GitHub 보안 기능 활성화 가이드

**작성일**: 2026-01-24  
**대상 프로젝트**: Cognitive Seed Framework  
**작성자**: 누스양 (Manus AI)

---

## 개요

본 가이드는 GitHub 저장소의 보안 기능을 활성화하여 프로젝트의 보안 수준을 강화하는 방법을 안내합니다.

---

## 1. Vulnerability Alerts (의존성 취약점 알림) 활성화

### 1.1. 기능 설명

- 프로젝트의 의존성에서 발견된 보안 취약점을 자동으로 감지하고 알림
- GitHub Advisory Database와 연동하여 최신 취약점 정보 제공
- Dependabot이 자동으로 보안 패치 PR 생성 가능

### 1.2. 활성화 방법

1. **GitHub 저장소 접속**:
   - https://github.com/tjwlstj/cognitive-seed-framework

2. **Settings 메뉴 이동**:
   - 저장소 상단 메뉴에서 `Settings` 클릭

3. **Security 섹션 이동**:
   - 왼쪽 사이드바에서 `Code security and analysis` 클릭

4. **Dependency graph 확인**:
   - "Dependency graph" 섹션이 활성화되어 있는지 확인
   - 비활성화 상태라면 `Enable` 버튼 클릭

5. **Dependabot alerts 활성화**:
   - "Dependabot alerts" 섹션에서 `Enable` 버튼 클릭
   - 이메일 알림 설정 확인

6. **Dependabot security updates 활성화** (권장):
   - "Dependabot security updates" 섹션에서 `Enable` 버튼 클릭
   - 취약점 발견 시 자동으로 PR 생성

### 1.3. 확인 방법

```bash
# GitHub CLI를 사용한 확인
gh api repos/tjwlstj/cognitive-seed-framework/vulnerability-alerts
```

활성화 성공 시 `200 OK` 응답 또는 취약점 목록이 반환됩니다.

---

## 2. Secret Scanning (비밀키 스캔) 활성화

### 2.1. 기능 설명

- 코드 내에 실수로 커밋된 API 키, 비밀번호, 토큰 등을 자동 탐지
- 지원되는 비밀키 패턴: GitHub 토큰, AWS 키, Azure 키, Slack 토큰 등 200+ 종류
- 발견 즉시 알림 및 비활성화 권장

### 2.2. 활성화 방법

1. **GitHub 저장소 Settings 이동**:
   - https://github.com/tjwlstj/cognitive-seed-framework/settings

2. **Code security and analysis 섹션**:
   - "Secret scanning" 섹션 확인

3. **활성화**:
   - Public 저장소: 기본적으로 활성화됨
   - Private 저장소: GitHub Advanced Security 필요 (유료)

4. **Push protection 활성화** (권장):
   - "Push protection" 옵션 활성화
   - 비밀키가 포함된 커밋을 푸시 전에 차단

### 2.3. 확인 방법

- Settings → Code security and analysis → Secret scanning 상태 확인
- 또는 저장소의 Security 탭에서 "Secret scanning alerts" 확인

---

## 3. CodeQL Analysis (코드 분석) 확인

### 3.1. 현재 상태

✅ **이미 활성화됨** (2026-01-13)

### 3.2. 기능 설명

- 정적 코드 분석을 통해 보안 취약점 자동 탐지
- SQL 인젝션, XSS, 경로 탐색 등 다양한 취약점 패턴 검사
- Pull Request 시 자동 분석 실행

### 3.3. 확인 방법

1. **GitHub Actions 워크플로우 확인**:
   ```bash
   ls -la .github/workflows/
   ```

2. **CodeQL 분석 결과 확인**:
   - 저장소의 `Security` 탭 → `Code scanning` 섹션

---

## 4. Dependabot 설정 확인

### 4.1. 현재 상태

✅ **이미 활성화됨**

### 4.2. 설정 파일 확인

```bash
cat .github/dependabot.yml
```

### 4.3. 권장 설정

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "tjwlstj"
    labels:
      - "dependencies"
      - "security"
```

---

## 5. 보안 정책 문서 확인

### 5.1. SECURITY.md

✅ **이미 존재함**

프로젝트의 보안 정책과 취약점 보고 방법이 명시되어 있습니다.

### 5.2. 확인 방법

- 저장소의 `Security` 탭 → `Policy` 섹션
- 또는 직접 파일 확인: https://github.com/tjwlstj/cognitive-seed-framework/blob/main/SECURITY.md

---

## 6. 활성화 체크리스트

### 6.1. 즉시 실행 필요

- [ ] **Vulnerability Alerts 활성화**
  - [ ] Dependabot alerts 활성화
  - [ ] Dependabot security updates 활성화
  - [ ] 이메일 알림 설정 확인

### 6.2. 권장 사항

- [ ] **Secret Scanning 확인**
  - [ ] Public 저장소: 자동 활성화 확인
  - [ ] Push protection 활성화 (가능한 경우)

- [ ] **CodeQL 설정 검토**
  - [x] 이미 활성화됨
  - [ ] 분석 결과 정기 검토

- [ ] **Dependabot 설정 최적화**
  - [x] 기본 설정 활성화됨
  - [ ] 리뷰어 및 라벨 설정 추가

### 6.3. 장기 관리

- [ ] 주간 보안 알림 검토
- [ ] 월간 의존성 업데이트 검토
- [ ] 분기별 보안 정책 업데이트

---

## 7. 활성화 후 확인 사항

### 7.1. Vulnerability Alerts 테스트

1. Settings → Code security and analysis 확인
2. Security 탭에서 "Dependabot alerts" 섹션 확인
3. 기존 취약점이 있다면 알림 확인

### 7.2. 알림 설정 확인

1. GitHub 프로필 → Settings → Notifications
2. "Security alerts" 섹션에서 알림 방법 설정:
   - Email 알림
   - Web 알림
   - Mobile 알림 (GitHub 앱 사용 시)

---

## 8. 문제 해결

### 8.1. Vulnerability Alerts가 활성화되지 않는 경우

**원인**:
- Dependency graph가 비활성화되어 있음
- 저장소 권한 부족

**해결 방법**:
1. Dependency graph 먼저 활성화
2. 저장소 Admin 권한 확인
3. 몇 분 후 다시 시도

### 8.2. Secret Scanning을 사용할 수 없는 경우

**원인**:
- Private 저장소에서 GitHub Advanced Security 미구독

**해결 방법**:
- Public 저장소로 전환 (가능한 경우)
- 또는 GitHub Advanced Security 구독 고려
- 로컬에서 git-secrets 등 도구 사용

---

## 9. 추가 보안 강화 방법

### 9.1. Branch Protection Rules 설정

1. Settings → Branches → Add rule
2. 권장 설정:
   - Require pull request reviews before merging
   - Require status checks to pass before merging
   - Require branches to be up to date before merging

### 9.2. 2FA (Two-Factor Authentication) 활성화

- GitHub 계정 보안 강화를 위해 2FA 활성화 권장
- Settings → Password and authentication → Two-factor authentication

---

## 10. 참고 자료

- [GitHub Security Features](https://docs.github.com/en/code-security)
- [Dependabot Documentation](https://docs.github.com/en/code-security/dependabot)
- [Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [CodeQL](https://codeql.github.com/)

---

**다음 검토일**: 2026-02-24  
**담당자**: 프로젝트 관리자 (tjwlstj)
