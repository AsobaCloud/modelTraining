#!/usr/bin/env python3
"""
Tests for BabeSource scraper - TDD approach
Written first to define expected behavior
"""

import pytest
import requests_mock
from unittest.mock import Mock, patch, MagicMock
import json
import hashlib
from PIL import Image
import io

# Import the scraper (will fail initially - that's TDD!)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

class TestBabeSourceScraper:
    
    def test_extract_gallery_urls_from_listing(self):
        """Test extraction of gallery URLs from listing page"""
        # Arrange
        mock_html = '''
        <html>
            <a href="/galleries/blake-blossom-ar-porn-190435.html">Gallery 1</a>
            <a href="/galleries/blake-blossom-playboy-190264.html">Gallery 2</a>
            <a href="/other-page.html">Not a gallery</a>
        </html>
        '''
        
        from babesource_scraper import BabeSourceScraper
        scraper = BabeSourceScraper('test-bucket')
        
        # Act
        with requests_mock.Mocker() as m:
            m.get('https://babesource.com/filter-content/?pornstar=9252', text=mock_html)
            gallery_urls = scraper.extract_gallery_urls('https://babesource.com/filter-content/?pornstar=9252')
        
        # Assert
        assert len(gallery_urls) == 2
        assert 'https://babesource.com/galleries/blake-blossom-ar-porn-190435.html' in gallery_urls
        assert 'https://babesource.com/galleries/blake-blossom-playboy-190264.html' in gallery_urls
    
    def test_extract_gallery_id_from_page(self):
        """Test extraction of gallery ID from gallery page"""
        # Arrange
        mock_html = '''
        <html>
            <a href="https://media.babesource.com/galleries/687231342db80/01.jpg">Photo 1</a>
            <a href="https://media.babesource.com/galleries/687231342db80/02.jpg">Photo 2</a>
        </html>
        '''
        
        from babesource_scraper import BabeSourceScraper
        scraper = BabeSourceScraper('test-bucket')
        
        # Act
        with requests_mock.Mocker() as m:
            m.get('https://babesource.com/galleries/test-gallery.html', text=mock_html)
            gallery_id = scraper.extract_gallery_id('https://babesource.com/galleries/test-gallery.html')
        
        # Assert
        assert gallery_id == '687231342db80'
    
    def test_generate_image_urls_from_gallery_id(self):
        """Test generation of complete image URL set from gallery ID"""
        # Arrange
        from babesource_scraper import BabeSourceScraper
        scraper = BabeSourceScraper('test-bucket')
        gallery_id = '687231342db80'
        
        # Act
        image_urls = scraper.generate_image_urls(gallery_id, max_images=5)
        
        # Assert
        expected_urls = [
            'https://media.babesource.com/galleries/687231342db80/01.jpg',
            'https://media.babesource.com/galleries/687231342db80/02.jpg',
            'https://media.babesource.com/galleries/687231342db80/03.jpg',
            'https://media.babesource.com/galleries/687231342db80/04.jpg',
            'https://media.babesource.com/galleries/687231342db80/05.jpg'
        ]
        assert image_urls == expected_urls
    
    def test_validate_image_quality_valid_image(self):
        """Test image quality validation for valid high-res image"""
        # Arrange
        from babesource_scraper import BabeSourceScraper
        scraper = BabeSourceScraper('test-bucket')
        
        # Create a test image (800x600)
        img = Image.new('RGB', (800, 600), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        image_data = img_bytes.getvalue()
        
        # Act
        is_valid = scraper.validate_image_quality(image_data)
        
        # Assert
        assert is_valid == True
    
    def test_validate_image_quality_invalid_small_image(self):
        """Test image quality validation rejects small images"""
        # Arrange
        from babesource_scraper import BabeSourceScraper
        scraper = BabeSourceScraper('test-bucket')
        
        # Create a small test image (400x300)
        img = Image.new('RGB', (400, 300), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        image_data = img_bytes.getvalue()
        
        # Act
        is_valid = scraper.validate_image_quality(image_data)
        
        # Assert
        assert is_valid == False
    
    def test_download_and_validate_image_success(self):
        """Test successful image download and validation"""
        # Arrange
        from babesource_scraper import BabeSourceScraper
        scraper = BabeSourceScraper('test-bucket')
        
        # Create valid test image data
        img = Image.new('RGB', (800, 600), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        test_image_data = img_bytes.getvalue()
        
        # Act
        with requests_mock.Mocker() as m:
            m.get('https://media.babesource.com/galleries/test/01.jpg', content=test_image_data)
            result = scraper.download_and_validate_image('https://media.babesource.com/galleries/test/01.jpg')
        
        # Assert
        assert result is not None
        assert len(result) > 0  # Has image data
    
    def test_download_and_validate_image_invalid_quality(self):
        """Test image download with invalid quality rejection"""
        # Arrange
        from babesource_scraper import BabeSourceScraper
        scraper = BabeSourceScraper('test-bucket')
        
        # Create invalid (small) test image
        img = Image.new('RGB', (300, 200), color='green')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        test_image_data = img_bytes.getvalue()
        
        # Act
        with requests_mock.Mocker() as m:
            m.get('https://media.babesource.com/galleries/test/01.jpg', content=test_image_data)
            result = scraper.download_and_validate_image('https://media.babesource.com/galleries/test/01.jpg')
        
        # Assert
        assert result is None  # Should reject low quality
    
    def test_process_gallery_end_to_end(self):
        """Test complete gallery processing workflow"""
        # Arrange
        from babesource_scraper import BabeSourceScraper
        scraper = BabeSourceScraper('test-bucket')
        
        gallery_html = '''
        <html>
            <a href="https://media.babesource.com/galleries/test123/01.jpg">Photo 1</a>
        </html>
        '''
        
        # Valid test image
        img = Image.new('RGB', (800, 600), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        test_image_data = img_bytes.getvalue()
        
        # Mock S3 client
        with patch('boto3.client') as mock_boto3:
            mock_s3 = Mock()
            mock_boto3.return_value = mock_s3
            
            with requests_mock.Mocker() as m:
                # Mock gallery page
                m.get('https://babesource.com/galleries/test.html', text=gallery_html)
                # Mock image downloads
                m.get('https://media.babesource.com/galleries/test123/01.jpg', content=test_image_data)
                m.get('https://media.babesource.com/galleries/test123/02.jpg', content=test_image_data)
                
                # Act
                result = scraper.process_gallery('https://babesource.com/galleries/test.html', max_images=2)
        
        # Assert
        assert result['success_count'] >= 1
        assert result['gallery_id'] == 'test123'
        mock_s3.put_object.assert_called()  # Verify S3 upload happened
    
    @pytest.fixture
    def mock_s3_setup(self):
        """Setup mock S3 client for tests"""
        with patch('boto3.client') as mock_boto3:
            mock_s3 = Mock()
            mock_boto3.return_value = mock_s3
            yield mock_s3

if __name__ == '__main__':
    pytest.main([__file__, '-v'])