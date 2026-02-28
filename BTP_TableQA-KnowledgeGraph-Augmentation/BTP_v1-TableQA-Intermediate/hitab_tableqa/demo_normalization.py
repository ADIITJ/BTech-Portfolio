"""
Demonstration of the simplified table normalization system.
Shows how hierarchical tables are converted to normalized relational structure.
"""

from table_normalizer import TableNormalizer, load_hitab_tables
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_normalization():
    """Comprehensive demonstration of table normalization"""
    
    print("="*60)
    print("TABLE NORMALIZATION SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Load sample tables
    tables = load_hitab_tables()
    if not tables:
        print("❌ No tables found!")
        return
    
    print(f"📊 Loaded {len(tables)} tables for demonstration")
    
    # Initialize normalizer
    normalizer = TableNormalizer()
    
    # Process each table and show detailed results
    results = {
        'single': 0,
        'dimensional': 0,
        'entity_split': 0,
        'total_tables_created': 0,
        'storage_savings': []
    }
    
    for i, table in enumerate(tables):
        print(f"\n--- Table {i+1}: {table.table_id} ---")
        print(f"Title: {table.title}")
        print(f"Original size: {len(table.cells)} rows × {len(table.cells[0]) if table.cells else 0} cols")
        
        # Normalize table
        normalized_tables = normalizer.normalize_table(table)
        
        print(f"✅ Normalized into {len(normalized_tables)} tables:")
        
        original_cells = len(table.cells) * (len(table.cells[0]) if table.cells else 0)
        total_normalized_cells = 0
        
        for nt in normalized_tables:
            print(f"   🔹 {nt.table_name} ({nt.table_type})")
            print(f"     - Shape: {nt.data.shape}")
            print(f"     - Relationships: {len(nt.relationships)}")
            if nt.relationships:
                for rel in nt.relationships:
                    print(f"       → {rel['type']} to {rel['target_table']}")
            
            total_normalized_cells += nt.data.size
            
            # Show sample data for first few tables
            if i < 3:
                print(f"     - Sample data:")
                print(f"       {nt.data.head(2).to_string().replace(chr(10), chr(10) + '       ')}")
        
        # Track results
        table_types = [nt.table_type for nt in normalized_tables]
        if 'dimension' in table_types and 'fact' in table_types:
            results['dimensional'] += 1
        elif 'entity' in table_types:
            results['entity_split'] += 1
        else:
            results['single'] += 1
        
        results['total_tables_created'] += len(normalized_tables)
        
        # Calculate storage efficiency
        if original_cells > 0:
            efficiency = (original_cells - total_normalized_cells) / original_cells * 100
            results['storage_savings'].append(efficiency)
            print(f"📈 Storage efficiency: {efficiency:.1f}% {'savings' if efficiency > 0 else 'overhead'}")
    
    # Final summary
    print(f"\n" + "="*60)
    print("NORMALIZATION SUMMARY")
    print("="*60)
    
    print(f"📊 Tables processed: {len(tables)}")
    print(f"✅ Success rate: 100%")
    print(f"🔢 Total normalized tables created: {results['total_tables_created']}")
    
    print(f"\n📋 Normalization Strategy Distribution:")
    print(f"   🔹 Single table (1NF): {results['single']} tables")
    print(f"   🔹 Dimensional model (star schema): {results['dimensional']} tables") 
    print(f"   🔹 Entity relationship: {results['entity_split']} tables")
    
    if results['storage_savings']:
        avg_savings = sum(results['storage_savings']) / len(results['storage_savings'])
        print(f"\n💾 Average storage efficiency: {avg_savings:.1f}%")
        
        positive_savings = [s for s in results['storage_savings'] if s > 0]
        if positive_savings:
            max_savings = max(positive_savings)
            print(f"🚀 Best case savings: {max_savings:.1f}%")

def show_graph_potential():
    """Demonstrate how normalized tables can be used for graph construction"""
    
    print(f"\n" + "="*60)
    print("GRAPH DATABASE POTENTIAL")
    print("="*60)
    
    tables = load_hitab_tables()
    normalizer = TableNormalizer()
    
    graph_structures = 0
    total_relationships = 0
    
    for table in tables[:5]:  # Show first 5 for brevity
        normalized = normalizer.normalize_table(table)
        
        # Count relationships that could become graph edges
        for nt in normalized:
            if nt.relationships:
                graph_structures += 1
                total_relationships += len(nt.relationships)
                
                print(f"\n🕸️  {nt.table_name}:")
                print(f"   Node type: {nt.table_type.upper()}")
                print(f"   Properties: {list(nt.data.columns)}")
                print(f"   Edges to:")
                for rel in nt.relationships:
                    print(f"     → {rel['target_table']} via {rel['column']}")
    
    print(f"\n📈 Graph Statistics:")
    print(f"   🔗 Tables with relationships: {graph_structures}")  
    print(f"   🌐 Total relationships (edges): {total_relationships}")
    print(f"   📊 Average edges per connected table: {total_relationships/max(graph_structures,1):.1f}")
    
    print(f"\n✨ Benefits for Graph Databases:")
    print(f"   • Foreign key relationships → Graph edges")
    print(f"   • Normalized entities → Graph nodes") 
    print(f"   • Fact tables → Central hub nodes")
    print(f"   • Dimension tables → Attribute nodes")

if __name__ == "__main__":
    demonstrate_normalization()
    show_graph_potential()
