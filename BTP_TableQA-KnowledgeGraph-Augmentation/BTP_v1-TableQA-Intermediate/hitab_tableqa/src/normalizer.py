"""
Dataset download/reading and normalization utilities for HiTab tables.
"""
import subprocess
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Iterator
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# --- Dataset Download/Reading ---
def download_hitab_dataset(data_dir: str = "data/raw") -> None:
    """Download HiTab dataset from GitHub repository."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    repo_url = "https://github.com/microsoft/HiTab"
    try:
        if not (data_path / "HiTab").exists():
            logger.info(f"Cloning HiTab repository to {data_dir}")
            subprocess.run([
                "git", "clone", repo_url, str(data_path / "HiTab")
            ], check=True)
        else:
            logger.info("HiTab repository already exists")
        # Extract tables and samples directories
        hitab_path = data_path / "HiTab"
        tables_src = hitab_path / "dataset" / "tables"
        samples_src = hitab_path / "dataset" / "samples"
        if tables_src.exists():
            subprocess.run([
                "cp", "-r", str(tables_src), str(data_path / "tables")
            ], check=True)
        if samples_src.exists():
            subprocess.run([
                "cp", "-r", str(samples_src), str(data_path / "samples")
            ], check=True)
        logger.info("Dataset download and extraction completed")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download dataset: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        raise

@dataclass
class TableNode:
    text: str
    children: List['TableNode']
    span: Tuple[int, int]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TableNode':
        return cls(
            text=data.get('name', data.get('text', '')),  # Use 'name' field if available
            children=[cls.from_dict(child) for child in data.get('children_dict', data.get('children', []))],
            span=(data.get('start', 0), data.get('end', 0))
        )

@dataclass
class HiTabTable:
    table_id: str
    title: str
    cells: List[List[str]]
    merged_regions: List[Dict[str, Any]]
    top_root: Optional[TableNode]
    left_root: Optional[TableNode]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HiTabTable':
        # Convert data format: list of lists of dicts to list of lists of strings
        raw_data = data.get('data', [])
        cells = []
        for row in raw_data:
            cell_row = []
            for cell in row:
                if isinstance(cell, dict) and 'value' in cell:
                    cell_row.append(str(cell['value']))
                else:
                    cell_row.append(str(cell))
            cells.append(cell_row)
        
        return cls(
            table_id=data.get('table_id', ''),
            title=data.get('title', ''),
            cells=cells,
            merged_regions=data.get('merged_regions', []),
            top_root=TableNode.from_dict(data['top_root']) if 'top_root' in data else None,
            left_root=TableNode.from_dict(data['left_root']) if 'left_root' in data else None
        )

# --- Normalization Logic ---
@dataclass
class NormalizedTable:
    table_id: str
    table_name: str
    data: pd.DataFrame
    relationships: List[Dict[str, Any]]
    source_table: str
    hierarchy_info: Dict[str, Any]

class HiTabNormalizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def normalize_table(self, table: HiTabTable) -> List[NormalizedTable]:
        """
        Convert hierarchical table into multiple normalized tables following database normalization principles.
        """
        try:
            # Analyze table structure to identify normalization opportunities
            analysis = self._analyze_table_structure(table)
            
            if analysis['normalization_type'] == 'single':
                return self._create_single_normalized_table(table, analysis)
            elif analysis['normalization_type'] == 'hierarchical_split':
                return self._create_hierarchical_normalized_tables(table, analysis)
            elif analysis['normalization_type'] == 'dimensional':
                return self._create_dimensional_normalized_tables(table, analysis)
            else:
                return self._create_single_normalized_table(table, analysis)
                
        except Exception as e:
            self.logger.error(f"Normalization failed for table {table.table_id}: {e}")
            return self._create_fallback_table(table)

    def _analyze_table_structure(self, table: HiTabTable) -> Dict[str, Any]:
        """Analyze hierarchical structure to determine normalization strategy."""
        analysis = {
            'normalization_type': 'single',
            'entities': [],
            'dimensions': [],
            'measures': [],
            'hierarchies': {},
            'relationships': []
        }
        
        if not table.cells or len(table.cells) == 0:
            return analysis
            
        # Extract column information from hierarchy
        columns = self._extract_column_info(table)
        analysis['columns'] = columns
        
        # Detect dimensional structure (fact/dimension pattern)
        if self._is_dimensional_table(table, columns):
            analysis['normalization_type'] = 'dimensional'
            analysis['dimensions'] = self._identify_dimensions(table, columns)
            analysis['measures'] = self._identify_measures(table, columns)
            
        # Detect hierarchical splitting opportunities
        elif self._has_hierarchical_split_potential(table, columns):
            analysis['normalization_type'] = 'hierarchical_split'
            analysis['entities'] = self._identify_entities(table, columns)
            analysis['relationships'] = self._identify_relationships(table, columns)
            
        return analysis

    def _extract_column_info(self, table: HiTabTable) -> List[Dict[str, Any]]:
        """Extract detailed column information from hierarchy."""
        columns = []
        
        if table.top_root and table.top_root.children:
            for i, child in enumerate(table.top_root.children):
                col_info = {
                    'index': i,
                    'name': child.text,
                    'hierarchy_path': self._get_hierarchy_path(child),
                    'data_type': self._infer_data_type(table, i),
                    'unique_values': self._count_unique_values(table, i),
                    'null_percentage': self._calculate_null_percentage(table, i)
                }
                columns.append(col_info)
        else:
            # Fallback for tables without hierarchy
            data_width = len(table.cells[0]) if table.cells else 0
            for i in range(data_width):
                col_info = {
                    'index': i,
                    'name': f'col_{i}',
                    'hierarchy_path': [],
                    'data_type': self._infer_data_type(table, i),
                    'unique_values': self._count_unique_values(table, i),
                    'null_percentage': self._calculate_null_percentage(table, i)
                }
                columns.append(col_info)
                
        return columns

    def _is_dimensional_table(self, table: HiTabTable, columns: List[Dict]) -> bool:
        """Check if table follows dimensional modeling pattern (fact table with dimensions)."""
        if len(columns) < 3:
            return False
            
        # Look for mix of categorical (dimensions) and numerical (measures) columns
        categorical_cols = [col for col in columns if col['data_type'] in ['string', 'categorical']]
        numerical_cols = [col for col in columns if col['data_type'] in ['numeric', 'float', 'integer']]
        
        return len(categorical_cols) >= 1 and len(numerical_cols) >= 1

    def _has_hierarchical_split_potential(self, table: HiTabTable, columns: List[Dict]) -> bool:
        """Check if table can be split based on hierarchical relationships."""
        if len(columns) < 4:
            return False
            
        # Look for repeated patterns indicating entity relationships
        for col in columns:
            if col['unique_values'] < len(table.cells) * 0.3:  # High repetition
                return True
                
        return False

    def _identify_dimensions(self, table: HiTabTable, columns: List[Dict]) -> List[Dict[str, Any]]:
        """Identify dimensional attributes for normalization."""
        dimensions = []
        
        for col in columns:
            if col['data_type'] in ['string', 'categorical'] and col['unique_values'] < len(table.cells):
                # This column has repeated values, good dimension candidate
                dim_values = self._extract_unique_dimension_values(table, col['index'])
                
                dimension = {
                    'name': col['name'],
                    'column_index': col['index'],
                    'values': dim_values,
                    'table_name': f"{table.table_id}_dim_{col['name'].lower().replace(' ', '_')}"
                }
                dimensions.append(dimension)
                
        return dimensions

    def _identify_measures(self, table: HiTabTable, columns: List[Dict]) -> List[Dict[str, Any]]:
        """Identify measure columns (numerical facts)."""
        measures = []
        
        for col in columns:
            if col['data_type'] in ['numeric', 'float', 'integer']:
                measure = {
                    'name': col['name'],
                    'column_index': col['index'],
                    'data_type': col['data_type'],
                    'aggregation': 'SUM'  # Default aggregation
                }
                measures.append(measure)
                
        return measures

    def _identify_entities(self, table: HiTabTable, columns: List[Dict]) -> List[Dict[str, Any]]:
        """Identify entities for hierarchical normalization."""
        entities = []
        
        # Group related columns into entities
        primary_entity = {
            'name': 'main_entity',
            'table_name': f"{table.table_id}_main",
            'columns': []
        }
        
        lookup_entities = []
        
        for col in columns:
            if col['unique_values'] < len(table.cells) * 0.5:  # Repeated values
                # Create lookup entity
                lookup_entity = {
                    'name': f"{col['name']}_lookup",
                    'table_name': f"{table.table_id}_lookup_{col['name'].lower().replace(' ', '_')}",
                    'columns': [col],
                    'values': self._extract_unique_dimension_values(table, col['index'])
                }
                lookup_entities.append(lookup_entity)
            else:
                primary_entity['columns'].append(col)
                
        entities.append(primary_entity)
        entities.extend(lookup_entities)
        return entities

    def _identify_relationships(self, table: HiTabTable, columns: List[Dict]) -> List[Dict[str, Any]]:
        """Identify foreign key relationships between entities."""
        relationships = []
        
        for col in columns:
            if col['unique_values'] < len(table.cells) * 0.5:
                relationship = {
                    'foreign_key': f"{col['name'].lower().replace(' ', '_')}_id",
                    'references_table': f"{table.table_id}_lookup_{col['name'].lower().replace(' ', '_')}",
                    'references_column': 'id',
                    'source_column': col['name']
                }
                relationships.append(relationship)
                
        return relationships

    def _create_dimensional_normalized_tables(self, table: HiTabTable, analysis: Dict) -> List[NormalizedTable]:
        """Create normalized tables using dimensional modeling approach."""
        normalized_tables = []
        
        # Create dimension tables
        for dimension in analysis['dimensions']:
            dim_df = pd.DataFrame()
            dim_df['id'] = range(1, len(dimension['values']) + 1)
            dim_df[dimension['name']] = [val['value'] for val in dimension['values']]
            
            dim_table = NormalizedTable(
                table_id=dimension['table_name'],
                table_name=f"{dimension['name']} Dimension",
                data=dim_df,
                relationships=[],
                source_table=table.table_id,
                hierarchy_info={
                    'table_type': 'dimension',
                    'dimension_name': dimension['name'],
                    'source_hierarchy': analysis.get('hierarchies', {})
                }
            )
            normalized_tables.append(dim_table)
        
        # Create fact table
        fact_data = []
        columns = analysis['columns']
        
        for row_idx, row in enumerate(table.cells):
            fact_row = {}
            
            # Add dimension foreign keys
            for dimension in analysis['dimensions']:
                col_idx = dimension['column_index']
                if col_idx < len(row):
                    cell_value = row[col_idx]
                    # Find dimension ID for this value
                    dim_id = next((val['id'] for val in dimension['values'] if val['value'] == cell_value), None)
                    fact_row[f"{dimension['name'].lower().replace(' ', '_')}_id"] = dim_id
            
            # Add measures
            for measure in analysis['measures']:
                col_idx = measure['column_index']
                if col_idx < len(row):
                    fact_row[measure['name']] = row[col_idx]
                    
            fact_data.append(fact_row)
        
        fact_df = pd.DataFrame(fact_data)
        
        fact_table = NormalizedTable(
            table_id=f"{table.table_id}_fact",
            table_name=f"{table.title} - Fact Table",
            data=fact_df,
            relationships=[{
                'type': 'foreign_key',
                'target_table': dim['table_name'],
                'column': f"{dim['name'].lower().replace(' ', '_')}_id"
            } for dim in analysis['dimensions']],
            source_table=table.table_id,
            hierarchy_info={
                'table_type': 'fact',
                'measures': analysis['measures'],
                'source_hierarchy': analysis.get('hierarchies', {})
            }
        )
        normalized_tables.append(fact_table)
        
        return normalized_tables

    def _create_hierarchical_normalized_tables(self, table: HiTabTable, analysis: Dict) -> List[NormalizedTable]:
        """Create normalized tables by splitting hierarchical entities."""
        normalized_tables = []
        
        for entity in analysis['entities']:
            if entity['name'] == 'main_entity':
                # Create main entity table with foreign key references
                main_data = []
                columns = analysis['columns']
                
                for row_idx, row in enumerate(table.cells):
                    main_row = {'id': row_idx + 1}
                    
                    # Add foreign keys to lookup tables
                    for rel in analysis['relationships']:
                        source_col_idx = next((col['index'] for col in columns if col['name'] == rel['source_column']), None)
                        if source_col_idx is not None and source_col_idx < len(row):
                            # Find ID in lookup table
                            lookup_entity = next((e for e in analysis['entities'] if e['table_name'] == rel['references_table']), None)
                            if lookup_entity:
                                cell_value = row[source_col_idx]
                                lookup_id = next((val['id'] for val in lookup_entity['values'] if val['value'] == cell_value), None)
                                main_row[rel['foreign_key']] = lookup_id
                    
                    # Add remaining columns
                    for col in entity['columns']:
                        if col['index'] < len(row):
                            main_row[col['name']] = row[col['index']]
                            
                    main_data.append(main_row)
                
                main_df = pd.DataFrame(main_data)
                main_table = NormalizedTable(
                    table_id=entity['table_name'],
                    table_name=f"{table.title} - Main",
                    data=main_df,
                    relationships=analysis['relationships'],
                    source_table=table.table_id,
                    hierarchy_info={
                        'table_type': 'main_entity',
                        'source_hierarchy': analysis.get('hierarchies', {})
                    }
                )
                normalized_tables.append(main_table)
                
            else:
                # Create lookup tables
                lookup_df = pd.DataFrame()
                lookup_df['id'] = range(1, len(entity['values']) + 1)
                lookup_df[entity['columns'][0]['name']] = [val['value'] for val in entity['values']]
                
                lookup_table = NormalizedTable(
                    table_id=entity['table_name'],
                    table_name=f"{entity['columns'][0]['name']} Lookup",
                    data=lookup_df,
                    relationships=[],
                    source_table=table.table_id,
                    hierarchy_info={
                        'table_type': 'lookup',
                        'lookup_for': entity['columns'][0]['name'],
                        'source_hierarchy': analysis.get('hierarchies', {})
                    }
                )
                normalized_tables.append(lookup_table)
        
        return normalized_tables

    def _create_single_normalized_table(self, table: HiTabTable, analysis: Dict) -> List[NormalizedTable]:
        """Create a single normalized table (1NF) when complex normalization isn't beneficial."""
        columns = analysis.get('columns', [])
        
        if not columns:
            # Fallback column creation
            if table.top_root and table.top_root.children:
                column_names = [child.text for child in table.top_root.children]
            elif table.cells and len(table.cells) > 0:
                column_names = [f"col_{i}" for i in range(len(table.cells[0]))]
            else:
                column_names = []
        else:
            column_names = [col['name'] for col in columns]
        
        # Ensure column count matches data
        if table.cells:
            data_width = len(table.cells[0])
            if len(column_names) != data_width:
                if len(column_names) > data_width:
                    column_names = column_names[:data_width]
                else:
                    column_names.extend([f"col_{i}" for i in range(len(column_names), data_width)])
        
        df = pd.DataFrame(table.cells, columns=column_names) if table.cells else pd.DataFrame()
        
        norm_table = NormalizedTable(
            table_id=table.table_id,
            table_name=table.title,
            data=df,
            relationships=[],
            source_table=table.table_id,
            hierarchy_info={
                'table_type': 'single_normalized',
                'normalization_form': '1NF',
                'has_top_hierarchy': table.top_root is not None,
                'has_left_hierarchy': table.left_root is not None,
                'top_hierarchy_depth': self._get_hierarchy_depth(table.top_root) if table.top_root else 0,
                'left_hierarchy_depth': self._get_hierarchy_depth(table.left_root) if table.left_root else 0
            }
        )
        return [norm_table]

    def _create_fallback_table(self, table: HiTabTable) -> List[NormalizedTable]:
        """Create fallback table when normalization fails."""
        df = pd.DataFrame(table.cells) if table.cells else pd.DataFrame()
        
        fallback_table = NormalizedTable(
            table_id=f"{table.table_id}_fallback",
            table_name=f"{table.title} (Fallback)",
            data=df,
            relationships=[],
            source_table=table.table_id,
            hierarchy_info={'table_type': 'fallback', 'error': 'normalization_failed'}
        )
        return [fallback_table]

    # Helper methods for data analysis
    def _get_hierarchy_path(self, node: TableNode) -> List[str]:
        """Get full hierarchy path for a node."""
        # Simplified - could be expanded to traverse full path
        return [node.text]

    def _infer_data_type(self, table: HiTabTable, col_index: int) -> str:
        """Infer data type of a column."""
        if not table.cells or col_index >= len(table.cells[0]):
            return 'unknown'
            
        sample_values = [row[col_index] for row in table.cells[:min(10, len(table.cells))] if col_index < len(row)]
        
        numeric_count = 0
        for val in sample_values:
            try:
                float(str(val).replace(',', '').replace('$', '').replace('%', ''))
                numeric_count += 1
            except (ValueError, TypeError):
                pass
                
        if numeric_count / len(sample_values) > 0.7:
            return 'numeric'
        else:
            return 'string'

    def _count_unique_values(self, table: HiTabTable, col_index: int) -> int:
        """Count unique values in a column."""
        if not table.cells:
            return 0
            
        values = set()
        for row in table.cells:
            if col_index < len(row):
                values.add(row[col_index])
        return len(values)

    def _calculate_null_percentage(self, table: HiTabTable, col_index: int) -> float:
        """Calculate percentage of null/empty values in a column."""
        if not table.cells:
            return 0.0
            
        null_count = 0
        total_count = 0
        
        for row in table.cells:
            if col_index < len(row):
                total_count += 1
                if not row[col_index] or row[col_index] == '' or row[col_index] == 'none':
                    null_count += 1
                    
        return (null_count / total_count * 100) if total_count > 0 else 0.0

    def _extract_unique_dimension_values(self, table: HiTabTable, col_index: int) -> List[Dict]:
        """Extract unique values with IDs for dimension tables."""
        unique_values = {}
        for row in table.cells:
            if col_index < len(row):
                value = row[col_index]
                if value not in unique_values:
                    unique_values[value] = len(unique_values) + 1
        
        return [{'id': id_val, 'value': value} for value, id_val in unique_values.items()]
    
    def _get_hierarchy_depth(self, node: Optional[TableNode]) -> int:
        """Calculate the depth of a hierarchy tree."""
        if not node or not node.children:
            return 0
        return 1 + max(self._get_hierarchy_depth(child) for child in node.children)

    def normalize_tables(self, tables: List[HiTabTable]) -> List[NormalizedTable]:
        all_normalized = []
        for table in tables:
            all_normalized.extend(self.normalize_table(table))
        self.logger.info(f"Normalized {len(tables)} tables into {len(all_normalized)} simple tables")
        return all_normalized

class HiTabLoader:
    """Loader for HiTab dataset tables and QA samples."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.tables_dir = self.data_dir / "tables" / "hmt"  # Updated path
        self.samples_dir = self.data_dir / "samples"
        
    def parse_table(self, table_path: Path) -> HiTabTable:
        """Parse a single table JSON file."""
        try:
            with open(table_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return HiTabTable.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to parse table {table_path}: {e}")
            raise
            
    def load_table(self, table_id: str) -> Optional[HiTabTable]:
        """Load a specific table by ID."""
        table_path = self.tables_dir / f"{table_id}.json"
        if not table_path.exists():
            logger.warning(f"Table file not found: {table_path}")
            return None
        return self.parse_table(table_path)
        
    def iter_tables(self, limit: Optional[int] = None) -> Iterator[HiTabTable]:
        """Iterate over tables in the dataset with optional limit."""
        if not self.tables_dir.exists():
            logger.error(f"Tables directory not found: {self.tables_dir}")
            return
            
        count = 0
        for table_file in self.tables_dir.glob("*.json"):
            if limit and count >= limit:
                break
            try:
                yield self.parse_table(table_file)
                count += 1
            except Exception as e:
                logger.warning(f"Skipping invalid table file {table_file}: {e}")
                continue
