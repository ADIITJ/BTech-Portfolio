#!/usr/bin/env python3
"""
Test script for advanced database normalization capabilities.
"""

import logging
import sys
import os
from pathlib import Path
sys.path.append('src')

from normalizer import HiTabLoader, HiTabNormalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_advanced_normalization():
    """Test the new advanced normalization capabilities."""
    
    print("🔄 Advanced Database Normalization Test")
    print("=" * 80)
    
    # Initialize
    loader = HiTabLoader()
    normalizer = HiTabNormalizer()
    
    # Test on first 3 tables
    table_count = 0
    for table in loader.iter_tables(limit=3):
        table_count += 1
        table_id = f"table_{table_count}"
        table.table_id = table_id
        
        print(f"\n📊 TESTING TABLE {table_count}: {table_id}")
        print("-" * 60)
        print(f"📝 Title: {table.title}")
        print(f"📏 Original Size: {len(table.cells)} rows × {len(table.cells[0]) if table.cells else 0} columns")
        
        # Show original hierarchy
        if table.top_root and table.top_root.children:
            print(f"🌳 Top Hierarchy: {[child.text for child in table.top_root.children[:8]]}")
        if table.left_root and table.left_root.children:
            print(f"🌳 Left Hierarchy: {[child.text for child in table.left_root.children[:5]]}")
        
        # Perform normalization
        try:
            normalized_tables = normalizer.normalize_table(table)
            
            print(f"\n✅ NORMALIZATION RESULT: {len(normalized_tables)} table(s) created")
            
            total_rows = 0
            for i, norm_table in enumerate(normalized_tables):
                print(f"\n📋 Table {i+1}: {norm_table.table_name}")
                print(f"   🆔 ID: {norm_table.table_id}")
                print(f"   📊 Size: {norm_table.data.shape}")
                print(f"   🏷️  Type: {norm_table.hierarchy_info.get('table_type', 'unknown')}")
                
                if norm_table.data.shape[1] > 0:
                    columns_display = list(norm_table.data.columns)
                    if len(columns_display) > 5:
                        columns_display = columns_display[:5] + ["..."]
                    print(f"   📋 Columns: {columns_display}")
                
                # Show relationships
                if norm_table.relationships:
                    print(f"   🔗 Relationships: {len(norm_table.relationships)}")
                    for rel in norm_table.relationships[:3]:  # Show first 3
                        print(f"      - {rel}")
                
                # Show sample data
                if not norm_table.data.empty and norm_table.data.shape[0] > 0:
                    print(f"   📄 Sample Data (first 2 rows):")
                    sample = norm_table.data.head(2).to_string(index=False, max_colwidth=15)
                    for line in sample.split('\n')[:3]:  # Show headers + 2 rows
                        print(f"      {line}")
                
                total_rows += norm_table.data.shape[0]
            
            print(f"\n📈 Summary:")
            print(f"   - Original: 1 table with {len(table.cells)} rows")
            print(f"   - Normalized: {len(normalized_tables)} tables with {total_rows} total rows")
            
            # Calculate normalization efficiency
            original_cells = len(table.cells) * (len(table.cells[0]) if table.cells else 0)
            normalized_cells = sum(t.data.shape[0] * t.data.shape[1] for t in normalized_tables)
            
            if original_cells > 0:
                efficiency = (1 - normalized_cells / original_cells) * 100
                print(f"   - Storage efficiency: {efficiency:.1f}% reduction in cell count")
            
        except Exception as e:
            print(f"❌ NORMALIZATION FAILED: {e}")
            logger.exception("Detailed error information:")
        
        print("\n" + "=" * 80)
    
    print(f"\n🎉 Advanced normalization testing completed on {table_count} tables!")

def analyze_normalization_types():
    """Analyze what types of normalization are being applied."""
    
    print("\n🔍 NORMALIZATION TYPE ANALYSIS")
    print("=" * 60)
    
    loader = HiTabLoader()
    normalizer = HiTabNormalizer()
    
    normalization_stats = {
        'single': 0,
        'dimensional': 0,
        'hierarchical_split': 0,
        'fallback': 0
    }
    
    table_count = 0
    for table in loader.iter_tables(limit=10):
        table_count += 1
        table.table_id = f"analysis_table_{table_count}"
        
        try:
            normalized_tables = normalizer.normalize_table(table)
            
            if normalized_tables:
                table_type = normalized_tables[0].hierarchy_info.get('table_type', 'unknown')
                
                if len(normalized_tables) == 1 and table_type == 'single_normalized':
                    normalization_stats['single'] += 1
                elif any(t.hierarchy_info.get('table_type') == 'fact' for t in normalized_tables):
                    normalization_stats['dimensional'] += 1
                elif any(t.hierarchy_info.get('table_type') == 'main_entity' for t in normalized_tables):
                    normalization_stats['hierarchical_split'] += 1
                else:
                    normalization_stats['fallback'] += 1
                    
                print(f"Table {table_count}: {len(normalized_tables)} tables -> {table_type}")
                
        except Exception as e:
            normalization_stats['fallback'] += 1
            print(f"Table {table_count}: FAILED -> {str(e)[:50]}...")
    
    print(f"\n📊 NORMALIZATION TYPE DISTRIBUTION:")
    for norm_type, count in normalization_stats.items():
        percentage = (count / table_count * 100) if table_count > 0 else 0
        print(f"   {norm_type.capitalize():20} {count:3d} tables ({percentage:5.1f}%)")

if __name__ == "__main__":
    test_advanced_normalization()
    analyze_normalization_types()
