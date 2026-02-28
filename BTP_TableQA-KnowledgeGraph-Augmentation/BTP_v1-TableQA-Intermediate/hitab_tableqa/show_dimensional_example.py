#!/usr/bin/env python3
"""
Demonstration of dimensional normalization results.
"""

import sys
sys.path.append('src')

from normalizer import HiTabLoader, HiTabNormalizer

def show_dimensional_example():
    """Show a detailed example of dimensional normalization."""
    
    loader = HiTabLoader()
    normalizer = HiTabNormalizer()
    
    print("🎯 DIMENSIONAL NORMALIZATION EXAMPLE")
    print("=" * 70)
    
    # Find a table that gets dimensionally normalized
    table_count = 0
    for table in loader.iter_tables(limit=20):
        table_count += 1
        table.table_id = f"demo_table_{table_count}"
        
        try:
            normalized_tables = normalizer.normalize_table(table)
            
            # Check if this is dimensionally normalized
            has_fact = any(t.hierarchy_info.get('table_type') == 'fact' for t in normalized_tables)
            has_dimension = any(t.hierarchy_info.get('table_type') == 'dimension' for t in normalized_tables)
            
            if has_fact or has_dimension:
                print(f"\n📊 FOUND DIMENSIONAL EXAMPLE: {table.title}")
                print("-" * 70)
                print(f"📏 Original: {len(table.cells)} rows × {len(table.cells[0]) if table.cells else 0} columns")
                
                # Show original data structure
                if table.cells:
                    print(f"\n📋 ORIGINAL DATA (first 3 rows):")
                    for i, row in enumerate(table.cells[:3]):
                        row_display = [str(cell)[:12] + "..." if len(str(cell)) > 12 else str(cell) for cell in row[:6]]
                        print(f"   Row {i}: {row_display}")
                
                print(f"\n🔄 NORMALIZED INTO {len(normalized_tables)} TABLES:")
                
                for i, norm_table in enumerate(normalized_tables):
                    table_type = norm_table.hierarchy_info.get('table_type', 'unknown')
                    print(f"\n📋 Table {i+1}: {table_type.upper()}")
                    print(f"   🆔 ID: {norm_table.table_id}")
                    print(f"   📊 Size: {norm_table.data.shape}")
                    print(f"   📋 Columns: {list(norm_table.data.columns)}")
                    
                    if norm_table.relationships:
                        print(f"   🔗 Relationships:")
                        for rel in norm_table.relationships:
                            print(f"      - {rel}")
                    
                    # Show sample data
                    if not norm_table.data.empty:
                        print(f"   📄 Sample Data:")
                        sample = norm_table.data.head(3).to_string(index=False, max_colwidth=12)
                        for line in sample.split('\n')[:4]:  # Headers + 3 rows
                            print(f"      {line}")
                
                # Calculate space savings
                original_cells = len(table.cells) * (len(table.cells[0]) if table.cells else 0)
                normalized_cells = sum(t.data.shape[0] * t.data.shape[1] for t in normalized_tables)
                
                if original_cells > 0:
                    savings = (1 - normalized_cells / original_cells) * 100
                    print(f"\n💾 STORAGE ANALYSIS:")
                    print(f"   - Original: {original_cells} cells")
                    print(f"   - Normalized: {normalized_cells} cells")
                    print(f"   - Space savings: {savings:.1f}%")
                
                # Show normalization benefits
                print(f"\n✅ NORMALIZATION BENEFITS:")
                if has_dimension:
                    print("   - ✅ Eliminated data redundancy with dimension tables")
                    print("   - ✅ Improved data integrity with foreign key relationships")
                if has_fact:
                    print("   - ✅ Separated facts from dimensions for better analytics")
                    print("   - ✅ Optimized for OLAP queries and aggregations")
                
                return  # Stop at first dimensional example
                
        except Exception as e:
            print(f"   ❌ Table {table_count} failed: {e}")
            continue
    
    print("\n🔍 No dimensional normalization examples found in first 20 tables.")
    print("   Most tables may be simple enough for single-table normalization.")

if __name__ == "__main__":
    show_dimensional_example()
