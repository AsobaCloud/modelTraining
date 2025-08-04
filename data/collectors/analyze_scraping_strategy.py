#!/usr/bin/env python3
"""
Analysis tool to compare current scraping results vs improved strategy potential
"""

import boto3
import json
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from PIL import Image
import io

def analyze_current_dataset():
    """Analyze the current dataset quality by source"""
    s3 = boto3.client('s3', region_name='us-east-1')
    
    try:
        # Get comprehensive validation report
        response = s3.get_object(Bucket='flux-dev-nsfw', Key='comprehensive_validation_report.json')
        report = json.loads(response['Body'].read())
        
        print("CURRENT DATASET ANALYSIS")
        print("=" * 50)
        
        print(f"Total files: {report['summary']['total_files']}")
        print(f"Valid images: {report['summary']['valid_images']}")
        print(f"Invalid images: {report['summary']['invalid_images']}")
        print(f"Overall validation rate: {report['summary']['validation_rate']}")
        
        print("\nBY FILE PATTERN:")
        for pattern, count in report['file_patterns'].items():
            # Count valid images for this pattern
            valid_count = len([img for img in report['valid_images'] if img.get('pattern') == pattern])
            invalid_count = count - valid_count
            validation_rate = (valid_count / count * 100) if count > 0 else 0
            
            print(f"{pattern}:")
            print(f"  Total: {count}, Valid: {valid_count}, Invalid: {invalid_count}")
            print(f"  Validation rate: {validation_rate:.1f}%")
            
            # Analyze resolution patterns for invalid images
            if pattern == 'categorized':
                resolutions = defaultdict(int)
                for img in report['invalid_images']:
                    if img.get('pattern') == pattern:
                        res = f"{img.get('width', 0)}x{img.get('height', 0)}"
                        resolutions[res] += 1
                
                print(f"  Top invalid resolutions:")
                for res, count in sorted(resolutions.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {res}: {count} images")
        
        print("\nPROBLEM IDENTIFICATION:")
        print("- 'categorized' pattern: 0% valid (all thumbnails)")
        print("- Resolution pattern: 460px width dominant (thumbnails)")
        print("- Need to follow gallery links to get full-size images")
        
        return report
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        return None

def analyze_site_structure(test_url="https://pornpics.com/galleries/ebony/"):
    """Analyze a site's structure to understand how to get full-size images"""
    
    print("\nSITE STRUCTURE ANALYSIS")
    print("=" * 50)
    
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        print(f"Analyzing: {test_url}")
        
        # Get the category/listing page
        response = session.get(test_url, timeout=30)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print("\nGALLERY LINKS FOUND:")
        gallery_links = soup.select('a.rel-link, a[href*="gallery"], a[href*="pic"]')
        print(f"Found {len(gallery_links)} potential gallery links")
        
        # Analyze first few gallery links
        for i, link in enumerate(gallery_links[:3]):
            href = link.get('href')
            if href:
                gallery_url = urljoin(test_url, href)
                print(f"\nGallery {i+1}: {gallery_url}")
                
                try:
                    # Get gallery page
                    gallery_response = session.get(gallery_url, timeout=30)
                    gallery_soup = BeautifulSoup(gallery_response.content, 'html.parser')
                    
                    # Look for full-size images
                    full_images = gallery_soup.select('img.main-image, img[data-original], img.gallery-image')
                    print(f"  Full-size image elements: {len(full_images)}")
                    
                    # Check image URLs
                    for img in full_images[:2]:
                        img_url = img.get('src') or img.get('data-original') or img.get('data-src')
                        if img_url:
                            full_img_url = urljoin(gallery_url, img_url)
                            print(f"  Image URL: {full_img_url}")
                            
                            # Try to get image dimensions
                            try:
                                img_response = session.get(full_img_url, timeout=10)
                                if img_response.status_code == 200:
                                    img_data = img_response.content
                                    img_obj = Image.open(io.BytesIO(img_data))
                                    width, height = img_obj.size
                                    file_size_mb = len(img_data) / (1024 * 1024)
                                    print(f"    Resolution: {width}x{height}, Size: {file_size_mb:.2f}MB")
                                    
                                    if width >= 512 and height >= 512:
                                        print(f"    ✅ HIGH QUALITY - suitable for training")
                                    else:
                                        print(f"    ❌ LOW QUALITY - too small")
                            except:
                                print(f"    Could not fetch image")
                
                except Exception as e:
                    print(f"  Error analyzing gallery: {e}")
        
        print("\nTHUMBNAIL vs FULL-SIZE URL PATTERNS:")
        thumbnails = soup.select('img.thumb, img[src*="thumb"]')
        for thumb in thumbnails[:3]:
            thumb_url = thumb.get('src')
            if thumb_url:
                print(f"Thumbnail: {thumb_url}")
                # Try to convert to full size
                full_url = thumb_url.replace('/thumbs/', '/pics/').replace('_thumb.', '.')
                print(f"Potential full: {full_url}")
        
    except Exception as e:
        print(f"Error analyzing site structure: {e}")

def estimate_improvement_potential():
    """Estimate how many additional quality images we could get"""
    
    print("\nIMPROVEMENT POTENTIAL ANALYSIS")
    print("=" * 50)
    
    # Current stats
    current_valid = 314
    current_invalid_categorized = 738  # All the thumbnail images
    
    print(f"Current valid images: {current_valid}")
    print(f"Current invalid categorized images: {current_invalid_categorized}")
    
    # Estimate potential from improved scraping
    categories_with_thumbnails = 16  # From validation report
    avg_images_per_category = current_invalid_categorized // categories_with_thumbnails
    
    print(f"\nCategories with thumbnails: {categories_with_thumbnails}")
    print(f"Average thumbnails per category: {avg_images_per_category}")
    
    # Conservative estimates
    gallery_success_rate = 0.3  # 30% of galleries accessible
    image_quality_rate = 0.6    # 60% of full images meet quality standards
    
    potential_galleries = current_invalid_categorized * gallery_success_rate
    potential_quality_images = potential_galleries * image_quality_rate
    
    print(f"\nIMPROVEMENT ESTIMATES:")
    print(f"Potential accessible galleries: {potential_galleries:.0f}")
    print(f"Estimated quality images from improved scraping: {potential_quality_images:.0f}")
    print(f"Total potential dataset: {current_valid + potential_quality_images:.0f}")
    
    if (current_valid + potential_quality_images) >= 1000:
        print("✅ IMPROVED SCRAPING COULD REACH 1000+ IMAGE TARGET")
    else:
        remaining_needed = 1000 - (current_valid + potential_quality_images)
        print(f"❌ Still need ~{remaining_needed:.0f} more images after improvement")
        print("   Recommend combining with personal dataset or upscaling")
    
    return potential_quality_images

def main():
    """Run comprehensive analysis"""
    print("SCRAPING STRATEGY ANALYSIS")
    print("=" * 60)
    
    # Analyze current dataset
    current_report = analyze_current_dataset()
    
    # Analyze site structure (comment out if you don't want to make web requests)
    try:
        analyze_site_structure()
    except:
        print("\nSkipping site structure analysis (no internet or blocked)")
    
    # Estimate improvement potential
    potential = estimate_improvement_potential()
    
    print("\nRECOMMENDATIONS:")
    print("=" * 30)
    print("1. ✅ Implement improved scraper to follow gallery links")
    print("2. ✅ Target full-resolution images instead of thumbnails")
    print("3. ✅ Add quality validation (min 512x512, min file size)")
    print("4. ✅ Test on small scale first")
    
    if potential < 500:
        print("5. ⚠️  Also consider personal dataset approach as backup")
        print("6. ⚠️  Selective upscaling for best quality thumbnails")
    else:
        print("5. ✅ Improved scraping should provide sufficient data")

if __name__ == '__main__':
    main()