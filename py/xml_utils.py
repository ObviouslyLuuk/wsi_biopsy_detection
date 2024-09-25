from xml.etree import ElementTree as ET
from xml.dom import minidom
import numpy as np

class AnnotationType:
    RECTANGLE = 'Rectangle'
    POLYGON = 'Polygon'

class AnnotationGroup:
    WILD_TYPE = 'WildType'
    OVER_EXPRESSION = 'OverExpression'
    NULL_MUTATION = 'NullMutation'
    DOUBLE_CLONES = 'DoubleClones'
    NO_CONSENSUS = 'NoConsensus'
    EXCLUDE = 'exclude'

DEFAULT_COLOR = "#F4FA58"

            
def prettify(elem):
    """
    Returns a pretty-printed xml string for the Element.

    Parameters:
        elem: Element
            xml element

    Returns:
        string
            Pretty-printed xml string
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


###############################################################################
# CONVERSION
###############################################################################
def xml_to_dict(tree):
    """
    Converts an xml tree to a dictionary.

    Parameters:
        tree: ElementTree
            xml tree

    Returns:
        dictionary
            Dictionary representation of the xml tree
    """
    root = tree.getroot()
    return _xml_to_dict(root)

def _xml_to_dict(element):
    """
    Converts an xml element to a dictionary.

    Parameters:
        element: Element
            xml element

    Returns:
        dictionary
            Dictionary representation of the xml element
    """
    d = {element.tag: {}}
    if element.attrib:
        d[element.tag]["attrib"] = element.attrib
    if element.text:
        # d[element.tag].update({"text": element.text})
        pass
    children = list(element)
    if children:
        d[element.tag]["children"] = []
        for child in children:
            d[element.tag]["children"].append(_xml_to_dict(child))
    return d

def dict_to_xml(d):
    """
    Converts a dictionary to an xml tree.

    Parameters:
        d: dictionary
            Dictionary

    Returns:
        ElementTree
            xml tree representation of the dictionary
    """
    assert isinstance(d, dict) and len(d) == 1
    tag, body = next(iter(d.items()))
    root = ET.Element(tag)
    _dict_to_xml(body, root)
    return ET.ElementTree(root)

def _dict_to_xml(d, root):
    """
    Converts a dictionary to an xml element.

    Parameters:
        d: dictionary
            Dictionary
        root: Element
            xml element to add the dictionary to
    """
    if "attrib" in d:
        root.attrib = d["attrib"]
    if "text" in d:
        root.text = d["text"]
    if "children" in d:
        for child in d["children"]:
            tag, body = next(iter(child.items()))
            child_element = ET.SubElement(root, tag)
            _dict_to_xml(body, child_element)




###############################################################################
# TEMPLATES
###############################################################################
def get_xml_template_dict():
    """
    Returns an xml template dictionary.

    Returns:
        xml_template_dict: dictionary
            Xml template dictionary
    """
    xml_template_dict = {
        'ASAP_Annotations': {"children": [
                {'Annotations': {"children": []}},
                {'AnnotationGroups': {"children": []}}
            ]
        }
    }
    for group, color in zip([AnnotationGroup.WILD_TYPE, AnnotationGroup.OVER_EXPRESSION, AnnotationGroup.NULL_MUTATION, 
                             AnnotationGroup.DOUBLE_CLONES, AnnotationGroup.NO_CONSENSUS, AnnotationGroup.EXCLUDE], 
                             ['#64fe2e', '#aaaa00', '#0000ff', '#ff0000', DEFAULT_COLOR, '#000000']):
        xml_template_dict['ASAP_Annotations']['children'][1]['AnnotationGroups']['children'].append({
            'Group': {'attrib': {'Name': group, 'PartOfGroup': 'None', 'Color': color}, 'children': 
                      [{'Attibutes': {}}]
            }
        })
    return xml_template_dict


def get_annotation_dict(contour, annotation_type=AnnotationType.POLYGON, annotation_group="None", color=DEFAULT_COLOR, name=None):
    """
    Returns an annotation dictionary for a contour.

    Parameters:
        contour: numpy array
            Contour
        annotation_type: string
            Annotation type
        annotation_group: string
            Annotation group
        color: string
            Color of the annotation

    Returns:
        annotation_dict: dictionary
            Annotation dictionary
    """
    center = np.mean(contour, axis=0).astype(np.int32)
    if name is None:
        name = f'x{center[0]}y{center[1]}'
    annotation_dict = {'attrib': {
            'Name': name,
            'Type': annotation_type,
            'PartOfGroup': annotation_group,
            'Color': color,
        }, 
        'children': [{'Coordinates': {'children': []}}]
    }
    for i, (x, y) in enumerate(contour):
        annotation_dict['children'][0]['Coordinates']['children'].append({'Coordinate': {
                'attrib': {
                    'Order': str(i),
                    'X': str(x),
                    'Y': str(y)
                }
            }
        })
    return annotation_dict




###############################################################################
# SAVING
###############################################################################
def map_contour_to_slide_coordinates(contour, spacing, process_spacing, area_box):
    """Map contour, which has coordinates relative to the scanned area, with spacing, to coordinates relative to the slide.
    Doesn't correct for the cut_patch_margin, so set it to 0."""
    return contour * 4 * spacing + np.array(area_box[:2])[None,:] * 4 * process_spacing


def add_contour_to_xml_dict(xml_dict, mapped_contour, annotation_group="None"):
    """Add contour to xml_dict as an annotation."""
    annotation_dict = get_annotation_dict(mapped_contour, annotation_group=annotation_group)
    if not xml_dict:
        xml_dict = get_xml_template_dict()
    xml_dict['ASAP_Annotations']['children'][0]['Annotations']['children'].append({'Annotation': annotation_dict})
    return xml_dict




###############################################################################
# EXAMPLES
###############################################################################
"""
        <Annotation Name="Annotation 0" Type="Rectangle" PartOfGroup="NullMutation" Color="#000000">
            <Coordinates>
                <Coordinate Order="0" X="112178" Y="29089"/>
                <Coordinate Order="1" X="122881" Y="29089"/>
                <Coordinate Order="2" X="122881" Y="22312.0996"/>
                <Coordinate Order="3" X="112178" Y="22312.0996"/>
            </Coordinates>
        </Annotation>

        <Annotation Name="Annotation 3" Type="Polygon" PartOfGroup="NullMutation" Color="#F4FA58">
            <Coordinates>
                <Coordinate Order="0" X="117203.203" Y="23387.2188"/>
                <Coordinate Order="1" X="116342.508" Y="23862.0859"/>
                <Coordinate Order="2" X="115541.172" Y="24425.9902"/>
                ...
                <Coordinate Order="42" X="118835.562" Y="23179.4648"/>
                <Coordinate Order="43" X="118271.656" Y="23387.2188"/>
                <Coordinate Order="44" X="117737.43" Y="23387.2188"/>
            </Coordinates>
        </Annotation>
"""
"""
<ASAP_Annotations>
    <Annotations>
    </Annotations>
    <AnnotationGroups>
        <Group Name="WildType" PartOfGroup="None" Color="#64fe2e">
            <Attributes/>
        </Group>
        <Group Name="OverExpression" PartOfGroup="None" Color="#aaaa00">
            <Attributes/>
        </Group>
        <Group Name="NullMutation" PartOfGroup="None" Color="#0000ff">
            <Attributes/>
        </Group>
        <Group Name="DoubleClones" PartOfGroup="None" Color="#ff0000">
            <Attributes/>
        </Group>
        <Group Name="exclude" PartOfGroup="None" Color="#000000">
            <Attributes/>
        </Group>
    </AnnotationGroups>
</ASAP_Annotations>
"""

