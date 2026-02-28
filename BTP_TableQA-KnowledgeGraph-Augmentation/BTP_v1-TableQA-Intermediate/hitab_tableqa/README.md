# Table Normalization System

## 🎯 Purpose
Converts hierarchical tables into **true normalized relational database structure** following 1NF, 2NF, and 3NF principles. Perfect for building knowledge graphs and semantic networks.

## 🔥 Key Features

### ✅ True Database Normalization
- **1NF Compliance**: Eliminates repeating groups, ensures atomic values
- **2NF Compliance**: Removes partial dependencies on composite keys  
- **3NF Compliance**: Eliminates transitive dependencies
- **Dimensional Modeling**: Creates fact and dimension tables for analytical queries

### 🕸️ Graph-Ready Output
- **Primary Keys**: For unique node identification
- **Foreign Keys**: For relationship edges
- **Entity Tables**: Normalized lookup tables
- **Fact Tables**: Central measurement tables
- **Relationship Metadata**: Complete graph structure information

### 📊 Smart Analysis
- **Automatic Strategy Selection**: Chooses best normalization approach
- **Data Type Inference**: Identifies numeric vs categorical data
- **Pattern Recognition**: Detects dimensional vs entity patterns
- **Storage Optimization**: Reduces redundancy while maintaining structure

## 🚀 Quick Start

```python
from table_normalizer import TableNormalizer, load_hitab_tables

# Load tables
tables = load_hitab_tables()

# Initialize normalizer  
normalizer = TableNormalizer()

# Normalize table
normalized = normalizer.normalize_table(tables[0])

# Results: List of NormalizedTable objects
for table in normalized:
    print(f"{table.table_name} ({table.table_type}): {table.data.shape}")
    print(f"Relationships: {table.relationships}")
```

## 📋 Test Results

```
🔥 TABLE NORMALIZATION SYSTEM - COMPLETE TEST
============================================
TRUE DATABASE NORMALIZATION: 1NF → 2NF → 3NF
Perfect for graph-based knowledge representation!

📊 Processing Results:
• Total tables processed: 5
• Success rate: 100%
• Total normalized tables created: 11

🔄 Normalization Strategies:
• Dimensional: 3 tables (fact/dimension model)
• Single: 2 tables (simple normalization)

🕸️ GRAPH-READY STRUCTURE:
✓ Primary keys for unique identification
✓ Foreign keys for relationships  
✓ Separated dimensions for efficient queries
✓ Fact tables for measurements
✓ 1NF, 2NF, 3NF compliance
```

## 📁 Project Structure

```
hitab_tableqa/
├── table_normalizer.py      # Complete normalization system
├── test_complete_system.py  # Comprehensive test & demo
├── requirements.txt         # Dependencies
├── data/raw/tables/         # Sample HiTab tables
└── README.md               # This file
```

## 🏗️ Normalization Strategies

### 1. Dimensional Model
Creates **fact tables** with **dimension tables** for analytical queries:
- **Fact Table**: Central measurements with foreign keys
- **Dimension Tables**: Lookup tables with unique entities
- **Perfect for**: Sales data, metrics, time-series analysis

### 2. Entity Split Model  
Creates separate **entity tables** for normalized relationships:
- **Main Table**: Core data with foreign key references
- **Entity Tables**: Normalized lookup tables
- **Perfect for**: Complex relational data, multiple entities

### 3. Single Table Model
Creates **single normalized table** in 1NF:
- **Normalized Table**: Cleaned, atomic values with primary key
- **Perfect for**: Simple data, flat structures

## 🎯 Graph Construction Ready

The normalized output provides everything needed for graph construction:

```python
# Example: Build graph from normalized tables
for table in normalized_tables:
    if table.table_type == 'dimension':
        # Create entity nodes
        for _, row in table.data.iterrows():
            graph.add_node(f"{table.table_id}_{row['id']}", 
                          label=row['name'], 
                          type='entity')
    
    elif table.table_type == 'fact':
        # Create fact nodes and relationships
        for _, row in table.data.iterrows():
            fact_node = f"fact_{row['fact_id']}"
            graph.add_node(fact_node, type='fact')
            
            # Add relationships from foreign keys
            for rel in table.relationships:
                target_id = row[rel['column']]
                graph.add_edge(fact_node, f"{rel['target_table']}_{target_id}")
```

## 🚀 Ready for Production

This system successfully normalizes hierarchical tables into proper relational structure that can be directly used for:
- **Knowledge Graph Construction**
- **Semantic Network Building** 
- **Entity-Relationship Modeling**
- **Graph Database Population**
- **Property Graph Creation**

The normalization follows database theory principles while maintaining all relationship information needed for effective graph representation.
