# GitHub Security 기능 활성화 가이드

**작성일**: 2026-01-25  
**작성자**: 누스양 (Manus AI)  
**대상 저장소**: tjwlstj/cognitive-seed-framework

---

## 1. 개요

본 가이드는 GitHub 저장소의 보안 기능을 활성화하는 방법을 안내합니다. 이러한 기능은 **저장소 소유자**만 활성화할 수 있습니다.

### 1.1. 현재 상태

| 기능 | 상태 |
|---|---|
| Dependabot | ✅ 활성화됨 |
| CodeQL | ✅ 활성화됨 |
| Security Policy | ✅ 활성화됨 |
| **Vulnerability Alerts** | ❌ 비활성화 (활성화 필요) |
| **Secret Scanning** | 📋 미확인 (활성화 권장) |

---

## 2. Vulnerability Alerts 활성화

### 2.1. 기능 설명

**Dependabot Alerts**는 저장소의 의존성에서 발견된 보안 취약점을 자동으로 감지하고 알림을 보냅니다.

**주요 기능**:
- 의존성 취약점 자동 감지
- 이메일 및 GitHub 알림
- 취약점 상세 정보 제공
- 수정 버전 제안

### 2.2. 활성화 방법

#### 단계 1: 저장소 설정 페이지 접속

1. GitHub에서 저장소 페이지로 이동:
   ```
   https://github.com/tjwlstj/cognitive-seed-framework
   ```

2. 상단 메뉴에서 **Settings** 클릭

#### 단계 2: 보안 설정 페이지 접속

1. 왼쪽 사이드바에서 **Security** 섹션 찾기
2. **Code security and analysis** 클릭

#### 단계 3: Dependabot Alerts 활성화

1. **Dependency graph** 섹션 찾기
   - 이미 활성화되어 있어야 합니다 (Dependabot 사용 중이므로)
   - 비활성화되어 있다면 **Enable** 클릭

2. **Dependabot alerts** 섹션 찾기
   - **Enable** 버튼 클릭
   - 활성화되면 초록색 체크마크가 표시됩니다

3. **Dependabot security updates** 섹션 찾기 (선택 사항)
   - **Enable** 버튼 클릭
   - 이 기능은 취약점이 발견되면 자동으로 PR을 생성합니다

### 2.3. 알림 설정

#### 이메일 알림 설정

1. GitHub 프로필 → **Settings** → **Notifications**
2. **Dependabot alerts** 섹션에서 알림 방식 선택:
   - ✅ **Email**: 이메일로 알림 받기
   - ✅ **Web and Mobile**: GitHub 웹/모바일에서 알림 받기

#### 알림 빈도 설정

- **Real-time**: 취약점 발견 즉시 알림
- **Daily digest**: 하루 한 번 요약 알림
- **Weekly digest**: 일주일에 한 번 요약 알림

**권장 설정**: Real-time (즉시 대응 가능)

### 2.4. 확인 방법

활성화 후 다음 방법으로 확인할 수 있습니다:

1. 저장소 메인 페이지 → **Security** 탭
2. **Dependabot alerts** 섹션에서 현재 알림 확인
3. 취약점이 있다면 목록이 표시됩니다

---

## 3. Secret Scanning 활성화

### 3.1. 기능 설명

**Secret Scanning**은 코드에 실수로 커밋된 비밀키, API 토큰, 비밀번호 등을 자동으로 감지합니다.

**주요 기능**:
- API 키, 토큰, 비밀번호 자동 감지
- 200개 이상의 서비스 패턴 지원
- 실시간 알림
- 히스토리 스캔

### 3.2. 활성화 방법

#### 단계 1: 저장소 설정 페이지 접속

1. GitHub에서 저장소 페이지로 이동
2. 상단 메뉴에서 **Settings** 클릭

#### 단계 2: 보안 설정 페이지 접속

1. 왼쪽 사이드바에서 **Security** 섹션 찾기
2. **Code security and analysis** 클릭

#### 단계 3: Secret Scanning 활성화

1. **Secret scanning** 섹션 찾기
   - **Enable** 버튼 클릭
   - 활성화되면 초록색 체크마크가 표시됩니다

2. **Push protection** (선택 사항)
   - **Enable** 버튼 클릭
   - 이 기능은 비밀키를 포함한 커밋을 푸시하기 전에 차단합니다
   - **권장**: 활성화

### 3.3. 알림 설정

#### 이메일 알림 설정

1. GitHub 프로필 → **Settings** → **Notifications**
2. **Secret scanning alerts** 섹션에서 알림 방식 선택:
   - ✅ **Email**: 이메일로 알림 받기
   - ✅ **Web and Mobile**: GitHub 웹/모바일에서 알림 받기

### 3.4. 확인 방법

활성화 후 다음 방법으로 확인할 수 있습니다:

1. 저장소 메인 페이지 → **Security** 탭
2. **Secret scanning alerts** 섹션에서 현재 알림 확인
3. 비밀키가 발견되면 목록이 표시됩니다

---

## 4. 추가 권장 설정

### 4.1. Security Policy 업데이트

**SECURITY.md** 파일을 최신 상태로 유지하세요:

```markdown
# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it by emailing:
- **Email**: [your-email@example.com]

Please do NOT create a public GitHub issue for security vulnerabilities.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| 1.x.x   | :x:                |

## Security Update Policy

- **Critical vulnerabilities**: Patched within 24 hours
- **High vulnerabilities**: Patched within 7 days
- **Medium vulnerabilities**: Patched within 30 days
- **Low vulnerabilities**: Reviewed quarterly
```

### 4.2. Branch Protection Rules

보안 강화를 위해 브랜치 보호 규칙을 설정하세요:

1. 저장소 → **Settings** → **Branches**
2. **Add branch protection rule** 클릭
3. 다음 옵션 활성화:
   - ✅ **Require a pull request before merging**
   - ✅ **Require status checks to pass before merging**
   - ✅ **Require conversation resolution before merging**
   - ✅ **Do not allow bypassing the above settings**

### 4.3. Code Scanning Alerts

CodeQL이 이미 활성화되어 있으므로, 알림 설정을 확인하세요:

1. 저장소 → **Security** → **Code scanning**
2. 발견된 알림 검토 및 해결

---

## 5. 보안 대시보드 활용

### 5.1. Security Overview

저장소의 전체 보안 상태를 한눈에 확인:

1. 저장소 → **Security** 탭
2. 다음 섹션 확인:
   - **Dependabot alerts**: 의존성 취약점
   - **Code scanning alerts**: 코드 취약점
   - **Secret scanning alerts**: 비밀키 노출

### 5.2. Security Advisories

프로젝트에 영향을 미치는 보안 권고사항 확인:

1. 저장소 → **Security** → **Advisories**
2. 영향을 받는 의존성 및 권장 조치 확인

---

## 6. 자동화 및 모니터링

### 6.1. GitHub Actions 통합

보안 스캔을 CI/CD 파이프라인에 통합:

```yaml
name: Security Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # 매주 일요일 자정

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install pip-audit
      
      - name: Run pip-audit
        run: |
          pip-audit --desc
```

### 6.2. 정기 보안 검토

다음 일정으로 보안 상태를 정기적으로 검토하세요:

- **매주**: Dependabot alerts 확인 및 해결
- **매월**: 전체 보안 감사 수행 (pip-audit)
- **분기별**: 보안 정책 및 절차 검토

---

## 7. 문제 해결

### 7.1. Dependabot Alerts가 활성화되지 않는 경우

**원인**: Dependency graph가 비활성화되어 있을 수 있습니다.

**해결**:
1. Settings → Security → Code security and analysis
2. **Dependency graph** 섹션에서 **Enable** 클릭
3. 그 후 **Dependabot alerts** 활성화

### 7.2. Secret Scanning이 활성화되지 않는 경우

**원인**: 저장소가 private이고 GitHub Free 플랜을 사용 중일 수 있습니다.

**해결**:
- Secret Scanning은 public 저장소에서는 무료입니다
- Private 저장소에서는 GitHub Advanced Security가 필요합니다
- 또는 저장소를 public으로 변경하세요

### 7.3. 알림을 받지 못하는 경우

**원인**: 알림 설정이 올바르지 않을 수 있습니다.

**해결**:
1. GitHub 프로필 → Settings → Notifications
2. 이메일 주소가 올바른지 확인
3. Dependabot alerts 및 Secret scanning alerts 알림이 활성화되어 있는지 확인

---

## 8. 추가 리소스

### 8.1. 공식 문서

- [GitHub Security Features](https://docs.github.com/en/code-security)
- [Dependabot Alerts](https://docs.github.com/en/code-security/dependabot/dependabot-alerts)
- [Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [CodeQL](https://docs.github.com/en/code-security/code-scanning)

### 8.2. 모범 사례

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

## 9. 체크리스트

보안 기능 활성화 완료 여부를 확인하세요:

- [ ] Dependency graph 활성화 확인
- [ ] Dependabot alerts 활성화
- [ ] Dependabot security updates 활성화
- [ ] Secret scanning 활성화
- [ ] Secret scanning push protection 활성화
- [ ] 알림 설정 확인 (이메일/웹)
- [ ] Security Policy (SECURITY.md) 업데이트
- [ ] Branch protection rules 설정
- [ ] GitHub Actions 보안 스캔 통합

---

## 10. 결론

GitHub의 보안 기능을 활성화하면 프로젝트의 보안 수준을 크게 향상시킬 수 있습니다. 

**핵심 권장 사항**:
1. ✅ Dependabot alerts 활성화 (필수)
2. ✅ Secret scanning 활성화 (강력 권장)
3. ✅ 정기적인 보안 검토 수행

이러한 기능을 활용하여 Cognitive Seed Framework 프로젝트의 보안을 지속적으로 강화하세요.

---

**작성자**: 누스양 (Manus AI)  
**문의**: GitHub Issues 또는 Discussions  
**최종 업데이트**: 2026-01-25
