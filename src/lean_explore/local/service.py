# src/lean_explore/local/service.py

"""Provides a service class for local Lean data exploration.

This module defines the Service class, which offers methods to search,
retrieve by ID, and get dependencies for statement groups using local
data assets (SQLite database, FAISS index, and embedding models).
"""

import logging
import os
import pathlib
import time
from typing import List, Optional, Dict, Any

from sqlalchemy import create_engine, select, or_
from sqlalchemy.orm import sessionmaker, Session as SQLAlchemySessionType, joinedload
from sqlalchemy.exc import SQLAlchemyError, OperationalError # Added OperationalError

# Assuming SentenceTransformer and faiss.Index are types, not necessarily loaded here
# if search.py handles their loading and returns typed objects.
# We'll import the loader functions themselves.
from sentence_transformers import SentenceTransformer # For type hinting if needed
import faiss # For type hinting if needed

from lean_explore import defaults
from .search import (
    perform_search,
    load_embedding_model,
    load_faiss_assets
)
from lean_explore.shared.models.db import (
    StatementGroup,
    Declaration, # Imported for relationship access via StatementGroup
    StatementGroupDependency
)
from lean_explore.shared.models.api import (
    APISearchResultItem,
    APIPrimaryDeclarationInfo,
    APISearchResponse,
    APICitationsResponse
)

logger = logging.getLogger(__name__)


class Service:
    """A service for interacting with local Lean explore data.

    This service loads necessary data assets (embedding model, FAISS index,
    database connection) upon initialization using default paths and parameters
    derived from the active toolchain. It provides methods for searching
    statement groups, retrieving them by ID, and fetching their dependencies (citations).

    Attributes:
        embedding_model: The loaded sentence embedding model.
        faiss_index: The loaded FAISS index.
        text_chunk_id_map: A list mapping FAISS indices to text chunk IDs.
        engine: The SQLAlchemy engine for database connections.
        SessionLocal: The SQLAlchemy sessionmaker for creating sessions.
        default_faiss_k (int): Default number of FAISS neighbors to retrieve.
        default_pagerank_weight (float): Default weight for PageRank.
        default_text_relevance_weight (float): Default weight for text relevance.
        default_name_match_weight (float): Default weight for name matching.
        default_semantic_similarity_threshold (float): Default similarity threshold.
        default_results_limit (int): Default limit for search results.
        default_faiss_nprobe (int): Default nprobe for FAISS IVF indexes.
    """

    def __init__(self):
        """Initializes the Service by loading data assets and configurations.

        Ensures the user data directory for the active toolchain exists.
        Loads the embedding model, FAISS index, and sets up the database engine.
        Paths for data assets are sourced from `lean_explore.defaults`.

        Raises:
            FileNotFoundError: If essential data files (FAISS index, map, DB)
                               are not found at their expected locations for the
                               active toolchain. Users will be guided to use
                               `leanexplore data fetch`.
            RuntimeError: If the embedding model fails to load or if other
                          critical initialization steps (like database connection
                          after file checks) fail.
        """
        logger.info("Initializing local Service...")
        # Ensure the base user data directory exists.
        # The specific toolchain version directory is expected to be created by `leanexplore data fetch`.
        try:
            # defaults.LEAN_EXPLORE_USER_DATA_DIR is ~/.lean_explore/data
            # defaults.LEAN_EXPLORE_TOOLCHAINS_BASE_DIR is ~/.lean_explore/data/toolchains
            # The actual asset paths like defaults.DEFAULT_DB_PATH point into a versioned subdir.
            # We ensure the parent of the specific toolchain path exists,
            # which is LEAN_EXPLORE_TOOLCHAINS_BASE_DIR.
            defaults.LEAN_EXPLORE_TOOLCHAINS_BASE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"User toolchains base directory ensured: {defaults.LEAN_EXPLORE_TOOLCHAINS_BASE_DIR}"
            )
        except OSError as e:
            logger.error(
                f"Could not create user toolchains base directory "
                f"{defaults.LEAN_EXPLORE_TOOLCHAINS_BASE_DIR}: {e}"
            )
            # This is not necessarily fatal if all files are somehow already in place,
            # but it's a warning sign. The subsequent file checks will be more definitive.

        # Load embedding model
        logger.info(f"Loading embedding model: {defaults.DEFAULT_EMBEDDING_MODEL_NAME}")
        self.embedding_model: Optional[SentenceTransformer] = load_embedding_model(
            defaults.DEFAULT_EMBEDDING_MODEL_NAME
        )
        if self.embedding_model is None:
            # This error is critical and not related to 'data fetch' for this model.
            raise RuntimeError(
                f"Failed to load embedding model: {defaults.DEFAULT_EMBEDDING_MODEL_NAME}. "
                "Check model name and network connection if it's downloaded on the fly."
            )

        # Load FAISS assets
        # Paths from defaults now point to the versioned toolchain directory.
        faiss_index_path = defaults.DEFAULT_FAISS_INDEX_PATH
        faiss_map_path = defaults.DEFAULT_FAISS_MAP_PATH
        logger.info(f"Attempting to load FAISS assets: Index='{faiss_index_path}', Map='{faiss_map_path}'")

        faiss_assets = load_faiss_assets(
            str(faiss_index_path), str(faiss_map_path)
        )
        if faiss_assets[0] is None or faiss_assets[1] is None:
            # load_faiss_assets (in search.py) already logs detailed file not found errors.
            error_message = (
                "Failed to load critical FAISS assets (index or ID map).\n"
                "Expected at:\n"
                f"  Index path: {faiss_index_path}\n"
                f"  ID map path: {faiss_map_path}\n"
                "Please run 'leanexplore data fetch' to download or update the data toolchain."
            )
            logger.error(error_message)
            raise FileNotFoundError(error_message)
        self.faiss_index: faiss.Index = faiss_assets[0]
        self.text_chunk_id_map: List[str] = faiss_assets[1]
        logger.info("FAISS assets loaded successfully.")

        # Check for database file and initialize engine
        db_path = defaults.DEFAULT_DB_PATH
        db_url = defaults.DEFAULT_DB_URL
        logger.info(f"Initializing database engine. Expected DB path: {db_path}")

        # Explicitly check if the DB file exists if it's a file-based SQLite DB
        is_file_db = db_url.startswith("sqlite:///")
        if is_file_db and not db_path.exists():
            error_message = (
                f"Database file not found at the expected location: {db_path}\n"
                "Please run 'leanexplore data fetch' to download the data toolchain."
            )
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        try:
            self.engine = create_engine(db_url)
            self.SessionLocal: sessionmaker[SQLAlchemySessionType] = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )
            # Test connection
            with self.engine.connect() as conn:
                logger.info("Database connection successful.")
        except OperationalError as oe:
            guidance = "Please check your database configuration or connection parameters."
            if is_file_db: # If it's a file DB, and it existed, but connection failed
                guidance = (
                    f"The database file at '{db_path}' might be corrupted, inaccessible, or not a valid SQLite file. "
                    "Consider running 'leanexplore data fetch' to get a fresh copy."
                )
            logger.error(
                f"Failed to initialize database engine or connection to {db_url}: {oe}\n{guidance}"
            )
            # It's a runtime issue if the file existed but is problematic, or if it's a non-file DB error.
            raise RuntimeError(f"Database initialization failed: {oe}. {guidance}") from oe
        except Exception as e: # Catch other SQLAlchemy or unexpected errors during engine setup
            logger.error(f"Unexpected error during database engine initialization: {e}", exc_info=True)
            raise RuntimeError(f"Database initialization failed unexpectedly: {e}") from e

        # Store default search parameters from defaults module
        self.default_faiss_k: int = defaults.DEFAULT_FAISS_K
        self.default_pagerank_weight: float = defaults.DEFAULT_PAGERANK_WEIGHT
        self.default_text_relevance_weight: float = defaults.DEFAULT_TEXT_RELEVANCE_WEIGHT
        self.default_name_match_weight: float = defaults.DEFAULT_NAME_MATCH_WEIGHT
        self.default_semantic_similarity_threshold: float = (
            defaults.DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD
        )
        self.default_results_limit: int = defaults.DEFAULT_RESULTS_LIMIT
        self.default_faiss_nprobe: int = defaults.DEFAULT_FAISS_NPROBE

        logger.info("Local Service initialized successfully.")

    def _serialize_sg_to_api_item(self, sg_orm: StatementGroup) -> APISearchResultItem:
        """Converts a StatementGroup ORM object to an APISearchResultItem Pydantic model.

        Args:
            sg_orm: The SQLAlchemy StatementGroup object.

        Returns:
            An APISearchResultItem Pydantic model instance.
        """
        primary_decl_info = APIPrimaryDeclarationInfo(
            lean_name=sg_orm.primary_declaration.lean_name if sg_orm.primary_declaration else None
        )
        return APISearchResultItem(
            id=sg_orm.id,
            primary_declaration=primary_decl_info,
            source_file=sg_orm.source_file,
            range_start_line=sg_orm.range_start_line,
            display_statement_text=sg_orm.display_statement_text,
            statement_text=sg_orm.statement_text,
            docstring=sg_orm.docstring,
            informal_description=sg_orm.informal_description,
        )

    def search(
        self,
        query: str,
        package_filters: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> APISearchResponse:
        """Performs a local search for statement groups.

        Args:
            query: The search query string.
            package_filters: An optional list of package names to filter results by.
            limit: An optional limit on the number of results to return.
                   If None, defaults.DEFAULT_RESULTS_LIMIT is used.

        Returns:
            An APISearchResponse object containing search results and metadata.
        
        Raises:
            RuntimeError: If the service was not properly initialized (e.g. assets missing).
            Exception: Propagates exceptions from `perform_search`.
        """
        start_time = time.time()
        actual_limit = limit if limit is not None else self.default_results_limit

        if not self.embedding_model or not self.faiss_index or not self.text_chunk_id_map:
            # This check is a safeguard; __init__ should prevent service instantiation
            # if these assets are missing.
            logger.error("Search service assets not loaded. Service may not have initialized correctly.")
            raise RuntimeError("Search service assets not loaded. Please ensure data has been fetched.")

        with self.SessionLocal() as session:
            try:
                ranked_results_orm = perform_search(
                    session=session,
                    query_string=query,
                    model=self.embedding_model,
                    faiss_index=self.faiss_index,
                    text_chunk_id_map=self.text_chunk_id_map,
                    faiss_k=self.default_faiss_k,
                    pagerank_weight=self.default_pagerank_weight,
                    text_relevance_weight=self.default_text_relevance_weight,
                    name_match_weight=self.default_name_match_weight,
                    selected_packages=package_filters,
                    semantic_similarity_threshold=self.default_semantic_similarity_threshold,
                    faiss_nprobe=self.default_faiss_nprobe
                )
            except Exception as e: # Catch exceptions from perform_search
                logger.error(f"Error during perform_search execution: {e}", exc_info=True)
                # Re-raise to allow higher-level error handling if needed by the caller
                # (e.g., MCP server might want to return a specific error response)
                raise

        api_results = [
            self._serialize_sg_to_api_item(sg_obj) for sg_obj, _scores in ranked_results_orm
        ]

        final_results = api_results[:actual_limit]
        end_time = time.time()
        processing_time_ms = int((end_time - start_time) * 1000)

        return APISearchResponse(
            query=query,
            packages_applied=package_filters,
            results=final_results,
            count=len(final_results),
            total_candidates_considered=len(api_results), # Number before final limit
            processing_time_ms=processing_time_ms
        )

    def get_by_id(self, group_id: int) -> Optional[APISearchResultItem]:
        """Retrieves a specific statement group by its ID from local data.

        Args:
            group_id: The unique identifier of the statement group.

        Returns:
            An APISearchResultItem if found, otherwise None.
        """
        with self.SessionLocal() as session:
            try:
                stmt_group_orm = (
                    session.query(StatementGroup)
                    .options(joinedload(StatementGroup.primary_declaration))
                    .filter(StatementGroup.id == group_id)
                    .first()
                )
                if stmt_group_orm:
                    return self._serialize_sg_to_api_item(stmt_group_orm)
                return None
            except SQLAlchemyError as e:
                logger.error(f"Database error in get_by_id for group_id {group_id}: {e}", exc_info=True)
                # For a service method, returning None on DB error might be acceptable,
                # or raise a custom service-level exception.
                return None
            except Exception as e: # Catch any other unexpected errors
                logger.error(f"Unexpected error in get_by_id for group_id {group_id}: {e}", exc_info=True)
                return None


    def get_dependencies(self, group_id: int) -> Optional[APICitationsResponse]:
        """Retrieves citations for a specific statement group from local data.

        Citations are the statement groups that the specified group_id depends on.

        Args:
            group_id: The unique identifier of the statement group for which
                      to fetch citations.

        Returns:
            An APICitationsResponse object if the source group is found and has citations,
            or an APICitationsResponse with an empty list if no citations,
            otherwise None if the source group itself is not found or a DB error occurs.
        """
        with self.SessionLocal() as session:
            try:
                # Check if the source statement group exists
                source_group_exists = session.query(StatementGroup.id).filter(StatementGroup.id == group_id).first()
                if not source_group_exists:
                    logger.warning(f"Source statement group ID {group_id} not found for dependency lookup.")
                    return None # Source group does not exist

                # Query for statement groups that `group_id` depends on (citations)
                cited_target_groups_orm = (
                    session.query(StatementGroup)
                    .join(StatementGroupDependency, StatementGroup.id == StatementGroupDependency.target_statement_group_id)
                    .filter(StatementGroupDependency.source_statement_group_id == group_id)
                    .options(joinedload(StatementGroup.primary_declaration))
                    .all()
                )

                citations_api_items = [
                    self._serialize_sg_to_api_item(sg_orm) for sg_orm in cited_target_groups_orm
                ]

                return APICitationsResponse(
                    source_group_id=group_id,
                    citations=citations_api_items,
                    count=len(citations_api_items)
                )
            except SQLAlchemyError as e:
                logger.error(f"Database error in get_dependencies for group_id {group_id}: {e}", exc_info=True)
                return None
            except Exception as e: # Catch any other unexpected errors
                logger.error(f"Unexpected error in get_dependencies for group_id {group_id}: {e}", exc_info=True)
                return None