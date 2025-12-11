"""
SEED-C07 Boundary Detector 단위 테스트
"""

import pytest
import torch
from seeds.cellular.c07_boundary_detector import BoundaryDetector, create_boundary_detector


class TestBoundaryDetector:
    """Boundary Detector 테스트"""
    
    @pytest.fixture
    def detector(self):
        """테스트용 Boundary Detector 생성"""
        return BoundaryDetector(input_dim=128, num_boundary_levels=4)
    
    @pytest.fixture
    def sample_input(self):
        """테스트용 입력 데이터"""
        return torch.randn(4, 50, 128)
    
    def test_initialization(self, detector):
        """초기화 테스트"""
        assert detector.config.seed_id == "SEED-C07"
        assert detector.config.name == "Boundary Detector"
        assert detector.config.level == 2
        assert detector.config.category == "Pattern"
        assert detector.config.num_boundary_levels == 4
        
        # 컴포넌트 확인
        assert hasattr(detector, 'edge_detector')
        assert hasattr(detector, 'pattern_completer')
        assert hasattr(detector, 'context_integrator')
        assert hasattr(detector, 'level_classifiers')
        assert len(detector.level_classifiers) == 4
    
    def test_forward_basic(self, detector, sample_input):
        """기본 forward pass 테스트"""
        result = detector(sample_input)
        
        # 출력 형태 확인
        assert 'boundaries' in result
        assert 'confidence' in result
        assert result['boundaries'].shape == (4, 50, 4)  # [B, L, num_levels]
        assert result['confidence'].shape == (4, 50)  # [B, L]
        
        # 값 범위 확인 (확률이므로 0~1)
        assert torch.all(result['boundaries'] >= 0)
        assert torch.all(result['boundaries'] <= 1)
        assert torch.all(result['confidence'] >= 0)
        assert torch.all(result['confidence'] <= 1)
    
    def test_forward_with_mask(self, detector, sample_input):
        """마스크를 사용한 forward pass 테스트"""
        mask = torch.ones(4, 50)
        mask[:, 40:] = 0  # 마지막 10개 위치는 패딩
        
        result = detector(sample_input, mask=mask)
        
        assert result['boundaries'].shape == (4, 50, 4)
        assert result['confidence'].shape == (4, 50)
    
    def test_forward_with_details(self, detector, sample_input):
        """상세 정보 반환 테스트"""
        result = detector(sample_input, return_details=True)
        
        # 추가 정보 확인
        assert 'features' in result
        assert 'edge_features' in result
        assert 'pattern_features' in result
        assert 'context_features' in result
        
        assert result['features'].shape == (4, 50, 128)
        assert result['edge_features'].shape == (4, 50, 128)
        assert result['pattern_features'].shape == (4, 50, 128)
        assert result['context_features'].shape == (4, 50, 128)
    
    def test_hierarchical_constraints(self, detector, sample_input):
        """계층적 제약 테스트"""
        result = detector(sample_input)
        boundaries = result['boundaries']  # [B, L, num_levels]
        
        # 계층적 제약 확인: level_i >= level_(i+1)
        for i in range(3):  # 0, 1, 2
            # 하위 레벨이 상위 레벨보다 크거나 같아야 함
            assert torch.all(boundaries[:, :, i] >= boundaries[:, :, i+1] - 1e-5)
    
    def test_detect_boundaries(self, detector, sample_input):
        """경계 검출 (이진 마스크) 테스트"""
        # 특정 레벨 검출
        mask_level_0 = detector.detect_boundaries(sample_input, level=0)
        assert mask_level_0.shape == (4, 50)
        assert torch.all((mask_level_0 == 0) | (mask_level_0 == 1))
        
        # 모든 레벨 검출
        mask_all = detector.detect_boundaries(sample_input)
        assert mask_all.shape == (4, 50, 4)
        assert torch.all((mask_all == 0) | (mask_all == 1))
    
    def test_detect_boundaries_with_threshold(self, detector, sample_input):
        """임계값을 사용한 경계 검출 테스트"""
        # 낮은 임계값
        mask_low = detector.detect_boundaries(sample_input, level=0, threshold=0.3)
        
        # 높은 임계값
        mask_high = detector.detect_boundaries(sample_input, level=0, threshold=0.7)
        
        # 낮은 임계값이 더 많은 경계를 검출해야 함
        assert mask_low.sum() >= mask_high.sum()
    
    def test_get_boundary_segments(self, detector, sample_input):
        """세그먼트 추출 테스트"""
        segments = detector.get_boundary_segments(sample_input, level=0)
        
        # 배치 크기만큼 세그먼트 리스트 생성
        assert len(segments) == 4
        
        # 각 배치의 세그먼트 확인
        for batch_segments in segments:
            assert isinstance(batch_segments, list)
            
            # 세그먼트가 연속적이고 전체 시퀀스를 커버하는지 확인
            if len(batch_segments) > 0:
                assert batch_segments[0][0] == 0  # 첫 세그먼트는 0부터 시작
                assert batch_segments[-1][1] == 50  # 마지막 세그먼트는 50에서 끝
                
                # 세그먼트가 연속적인지 확인
                for i in range(len(batch_segments) - 1):
                    assert batch_segments[i][1] == batch_segments[i+1][0]
    
    def test_compute_boundary_metrics(self, detector):
        """경계 검출 성능 평가 테스트"""
        # 가짜 예측과 정답 생성
        predicted = torch.rand(4, 50, 4)
        ground_truth = torch.zeros(4, 50, 4)
        ground_truth[:, [10, 20, 30, 40], :] = 1.0  # 특정 위치에 경계 설정
        
        metrics = detector.compute_boundary_metrics(predicted, ground_truth)
        
        # 메트릭 키 확인
        assert 'level_0_precision' in metrics
        assert 'level_0_recall' in metrics
        assert 'level_0_f1' in metrics
        assert 'avg_precision' in metrics
        assert 'avg_recall' in metrics
        assert 'avg_f1' in metrics
        
        # 값 범위 확인 (0~1)
        for key, value in metrics.items():
            assert 0 <= value <= 1
    
    def test_different_input_sizes(self, detector):
        """다양한 입력 크기 테스트"""
        # 짧은 시퀀스
        x_short = torch.randn(2, 10, 128)
        result_short = detector(x_short)
        assert result_short['boundaries'].shape == (2, 10, 4)
        
        # 긴 시퀀스
        x_long = torch.randn(2, 200, 128)
        result_long = detector(x_long)
        assert result_long['boundaries'].shape == (2, 200, 4)
    
    def test_batch_size_one(self, detector):
        """배치 크기 1 테스트"""
        x = torch.randn(1, 50, 128)
        result = detector(x)
        
        assert result['boundaries'].shape == (1, 50, 4)
        assert result['confidence'].shape == (1, 50)
    
    def test_create_boundary_detector(self):
        """생성 함수 테스트"""
        detector = create_boundary_detector(input_dim=256, num_boundary_levels=3)
        
        assert detector.config.input_dim == 256
        assert detector.config.num_boundary_levels == 3
        assert len(detector.level_classifiers) == 3
    
    def test_gradient_flow(self, detector, sample_input):
        """그래디언트 흐름 테스트"""
        sample_input.requires_grad = True
        
        result = detector(sample_input)
        loss = result['boundaries'].sum() + result['confidence'].sum()
        loss.backward()
        
        # 그래디언트가 입력까지 전파되는지 확인
        assert sample_input.grad is not None
        assert not torch.all(sample_input.grad == 0)
    
    def test_deterministic_output(self, detector, sample_input):
        """결정론적 출력 테스트"""
        detector.eval()
        
        with torch.no_grad():
            result1 = detector(sample_input)
            result2 = detector(sample_input)
        
        # 같은 입력에 대해 같은 출력
        assert torch.allclose(result1['boundaries'], result2['boundaries'])
        assert torch.allclose(result1['confidence'], result2['confidence'])
    
    def test_device_compatibility(self, detector, sample_input):
        """디바이스 호환성 테스트"""
        if torch.cuda.is_available():
            detector_cuda = detector.cuda()
            input_cuda = sample_input.cuda()
            
            result = detector_cuda(input_cuda)
            
            assert result['boundaries'].device.type == 'cuda'
            assert result['confidence'].device.type == 'cuda'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
